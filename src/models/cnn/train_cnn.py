"""
Phase 5 (Stage 1) — Image-only CNN, transfer learning (ResNet-18).

Design decisions, stated explicitly:
  - ResNet-18 pretrained on ImageNet, fc layer replaced for binary output.
    Chosen for speed on a single RTX 4050 laptop GPU within today's time
    budget — not the highest-capacity option, but a reasonable, fast
    baseline; swap in EfficientNet/DenseNet later if time allows.
  - Per-patient inverse-frequency WeightedRandomSampler at the image level,
    so a patient with 28 images doesn't dominate the loss relative to a
    patient with 1 — same rationale as the RICORD per-patient weighting
    decision, applied here too since Cov-Caldas has the same imbalance
    (mean 3.11 images/patient, max 28).
  - Class-weighted loss (pos_weight = neg_count / pos_count in TRAIN),
    consistent with the ~3.5x decided for the 22:78 class imbalance.
  - No horizontal flip augmentation. CXRs are not left-right symmetric
    (heart position, gastric bubble) and some sources embed laterality
    markers in-frame — flipping risks training on a distribution the model
    won't see at test time, for no proven benefit here. Mild rotation +
    brightness/contrast jitter only.
  - Evaluated on FOUR sets: full val, full test, age-matched val, age-matched
    test (from build_age_matched_eval_set.py). Report all four — the matched
    numbers are the real shortcut-learning stress test given the Phase 4
    metadata ceiling (RF test AUC 0.938, age = 97% of importance).

Run from src/models/cnn/. Requires torch + torchvision
(pip install torch torchvision --break-system-packages, or per your CUDA
version — see pytorch.org for the right command for your GPU).
"""

import time
import warnings
warnings.filterwarnings("ignore", message="A NumPy version")

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def find_project_root():
    """Walk upward until a folder containing data/processed is found, so the
    script works from any depth instead of assuming a fixed number of parents."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "models" / "checkpoints"
REPORTS = ROOT / "reports" / "baselines"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src" / "utils"))
from image_io import load_normalized_image


CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "lr": 1e-4,
    "max_epochs": 15,
    "patience": 4,
    "num_workers": 2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- dataset --

class CovCaldasImageDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].values
        self.labels = (df["stage1_label"] == "positive").astype(int).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_normalized_image(self.paths[idx])
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.RandomRotation(7),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ------------------------------------------------------------- data setup --

def make_train_sampler(train_df):
    """Per-patient inverse-frequency weighting: each image's weight is
    1 / (number of images that patient has), so every patient contributes
    equally in expectation regardless of how many images they have."""
    images_per_patient = train_df.groupby("patient_id")["patient_id"].transform("count")
    weights = 1.0 / images_per_patient
    return WeightedRandomSampler(weights.values, num_samples=len(train_df), replacement=True)


def load_splits():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    val_matched_path = PROCESSED / "cov_caldas_val_age_matched.csv"
    test_matched_path = PROCESSED / "cov_caldas_test_age_matched.csv"
    val_matched_df = pd.read_csv(val_matched_path) if val_matched_path.exists() else None
    test_matched_df = pd.read_csv(test_matched_path) if test_matched_path.exists() else None

    if val_matched_df is None or test_matched_df is None:
        print("[FLAG] Age-matched eval sets not found — run build_age_matched_eval_set.py "
              "first (src/evaluation/) for the full shortcut-learning comparison. "
              "Continuing with full val/test only.")

    return train_df, val_df, test_df, val_matched_df, test_matched_df


# ------------------------------------------------------------------ model --

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)


# --------------------------------------------------------------- training --

def run_epoch(model, loader, criterion, optimizer=None, progress=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch_idx, (imgs, labels) in enumerate(loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if progress and (batch_idx % 20 == 0 or batch_idx == len(loader)):
                seen = min(batch_idx * loader.batch_size, len(loader.dataset))
                print(f"    [{progress}] batch {batch_idx}/{len(loader)}  "
                      f"running loss={total_loss / seen:.4f}", flush=True)

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc, np.array(all_probs), np.array(all_labels)


def evaluate_set(model, df, name, criterion):
    if df is None or len(df) == 0:
        print(f"  {name}: skipped (no data)")
        return None

    if df["stage1_label"].nunique() < 2:
        print(f"  {name}: skipped (only one class present, AUC undefined)")
        return None

    loader = DataLoader(
        CovCaldasImageDataset(df, eval_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    loss, auc, probs, labels = run_epoch(
        model, loader, criterion, optimizer=None
    )

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print(
        f"  {name}: AUC={auc:.4f}  Acc={acc:.4f}  "
        f"Sens={sens:.4f}  Spec={spec:.4f}  (n={len(df)})"
    )

    return {
        "set": name,
        "auc": auc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "n": len(df)
    }


def main():
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cpu":
        print("[WARNING] No CUDA GPU detected — this will be slow on CPU. "
              "Check your torch install matches your CUDA version if you have an RTX 4050.")

    train_df, val_df, test_df, val_matched_df, test_matched_df = load_splits()

    print(
        f"Train: {len(train_df)} images "
        f"({train_df['patient_id'].nunique()} patients)"
    )
    print(f"Val:   {len(val_df)} images | Test: {len(test_df)} images")

    pos_weight_value = (
        (train_df["stage1_label"] == "negative").sum()
        / (train_df["stage1_label"] == "positive").sum()
    )

    print(f"pos_weight (class-weighted loss): {pos_weight_value:.2f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            float(pos_weight_value),
            dtype=torch.float32,
            device=DEVICE
        )
    )

    train_loader = DataLoader(
        CovCaldasImageDataset(train_df, train_transform),
        batch_size=CONFIG["batch_size"],
        sampler=make_train_sampler(train_df),
        num_workers=CONFIG["num_workers"],
    )

    val_loader = DataLoader(
        CovCaldasImageDataset(val_df, eval_transform),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    model = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"]
    )

    best_val_auc = -1.0
    epochs_without_improvement = 0

    best_ckpt_path = CHECKPOINTS / "cov_caldas_resnet18_best.pt"

    print(
        f"\n=== Training (max {CONFIG['max_epochs']} epochs, "
        f"patience {CONFIG['patience']}) ==="
    )

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        t0 = time.time()

        print(
            f"--- Epoch {epoch}/{CONFIG['max_epochs']} ---",
            flush=True
        )

        train_loss, train_auc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            progress="train"
        )

        val_loss, val_auc, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            progress="val"
        )

        print(
            f"Epoch {epoch:2d}: "
            f"train_loss={train_loss:.4f} "
            f"train_auc={train_auc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} | "
            f"{time.time() - t0:.0f}s",
            flush=True
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_ckpt_path
            )

            print(
                f"  -> new best val AUC, checkpoint saved to "
                f"{best_ckpt_path}"
            )

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= CONFIG["patience"]:
                print(
                    f"  -> no improvement for "
                    f"{CONFIG['patience']} epochs, stopping early."
                )
                break

    print(
        f"\n=== Loading best checkpoint "
        f"(val AUC={best_val_auc:.4f}) for final evaluation ==="
    )

    model.load_state_dict(
        torch.load(
            best_ckpt_path,
            map_location=DEVICE
        )
    )

    print("\n=== Final evaluation ===")

    results = []

    results.append(
        evaluate_set(
            model,
            val_df,
            "val (full)",
            criterion
        )
    )

    results.append(
        evaluate_set(
            model,
            test_df,
            "test (full)",
            criterion
        )
    )

    results.append(
        evaluate_set(
            model,
            val_matched_df,
            "val (age-matched)",
            criterion
        )
    )

    results.append(
        evaluate_set(
            model,
            test_matched_df,
            "test (age-matched)",
            criterion
        )
    )

    results_df = pd.DataFrame(
        [r for r in results if r is not None]
    )

    results_df.to_csv(
        REPORTS / "cov_caldas_cnn_results.csv",
        index=False
    )

    print(
        f"\nSaved results to "
        f"{REPORTS / 'cov_caldas_cnn_results.csv'}"
    )

    metadata_path = (
        REPORTS /
        "cov_caldas_metadata_baseline_results.csv"
    )

    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        meta_test = meta_df[
            meta_df["split"] == "test"
        ]

        print(
            "\n=== Shortcut-learning comparison: "
            "CNN vs. metadata-only baseline (test set) ==="
        )

        print(
            meta_test[["model", "auc"]].to_string(
                index=False
            )
        )

        full_test_row = next(
            (
                r for r in results
                if r and r["set"] == "test (full)"
            ),
            None
        )

        matched_test_row = next(
            (
                r for r in results
                if r and r["set"] == "test (age-matched)"
            ),
            None
        )

        if full_test_row:
            print(
                f"CNN (full test):         "
                f"AUC={full_test_row['auc']:.4f}"
            )

        if matched_test_row:
            print(
                f"CNN (age-matched test):  "
                f"AUC={matched_test_row['auc']:.4f}"
            )

            best_meta_auc = meta_test["auc"].max()

            if matched_test_row["auc"] < best_meta_auc + 0.03:
                print(
                    f"[FLAG] CNN's age-matched AUC "
                    f"({matched_test_row['auc']:.4f}) is close to or "
                    f"below the best metadata-only AUC "
                    f"({best_meta_auc:.4f}) — weak evidence "
                    f"against shortcut learning. Grad-CAM becomes "
                    f"essential here."
                )
            else:
                print(
                    f"[GOOD] CNN's age-matched AUC "
                    f"({matched_test_row['auc']:.4f}) clears the "
                    f"metadata ceiling "
                    f"({best_meta_auc:.4f}) meaningfully — evidence "
                    f"it's using signal beyond age/sex. Still report "
                    f"Grad-CAM as corroborating evidence."
                )

    else:
        print(
            "\n[FLAG] No metadata baseline results found — "
            "run baseline_metadata.py first "
            "for the shortcut-learning comparison."
        )


if __name__ == "__main__":
    main()