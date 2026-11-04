"""
Phase 5, step 5.4 - Grad-CAM for the Stage 1 ResNet-18.

Purpose: a qualitative check of WHERE the model looks. Given the age
confound (Phase 4: age+sex alone reach test AUC ~0.94), attention on lung
fields is consistent with a real signal, while attention on bone edges,
image borders, text/markers, or soft tissue outside the lungs would be a
red flag for shortcut learning.

Honest limits:
  - Grad-CAM is a visual aid, not proof. Looking at lungs does not prove the
    model ignores age cues, and looking elsewhere does not prove it cheats.
  - Images are a RANDOM sample (fixed seed) per true class - correct AND
    wrong predictions are shown. No cherry-picking of good-looking cases.
  - The map is always for the POSITIVE-class logit. For a true-negative
    image it shows what the model would count as evidence for "positive".

Uses the checkpoint written by train_cnn.py. Prefers the age-matched test
set if it exists (that is the less-confounded set), else the full test set.
Run from src/models/cnn/. Needs matplotlib (pip install matplotlib).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
from image_io import set_normalization_stats, load_normalized_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"
CKPT = ROOT / "models" / "checkpoints" / "cov_caldas_resnet18_best.pt"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224          # must match train_cnn.py
N_PER_CLASS = 6
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(CKPT, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        stats = checkpoint.get("normalization_stats")
        if stats is not None:
            set_normalization_stats(*stats)
    else:
        model.load_state_dict(checkpoint)
        raise RuntimeError("This legacy checkpoint has no frozen normalization statistics; retrain with the current train_cnn.py before generating Grad-CAM.")
    return model.to(DEVICE).eval()


def grad_cam(model, x):
    """Returns (cam[H,W] in 0..1, positive-class probability)."""
    store = {}
    layer = model.layer4[-1]

    def fwd_hook(_, __, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        logit = model(x)[0, 0]
        logit.backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.cpu().numpy(), torch.sigmoid(logit.detach()).item()


def pick_samples(df):
    rng = np.random.RandomState(SEED)
    picks = []
    for label in ["positive", "negative"]:
        sub = df[df["stage1_label"] == label]
        n = min(N_PER_CLASS, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
    return pd.concat(picks)


def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}. Run train_cnn.py first.")

    matched = PROCESSED / "cov_caldas_test_age_matched_FIXED.csv"
    if matched.exists():
        df, tag = pd.read_csv(matched), "test_age_matched"
    else:
        full = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
        df, tag = full[full["split"] == "test"], "test_full"
        print("[FLAG] Age-matched test set not found - using the full test set.")
    print(f"Grad-CAM source: {tag} ({len(df)} images)")

    model = load_model()
    samples = pick_samples(df)

    fig, axes = plt.subplots(2, N_PER_CLASS, figsize=(3 * N_PER_CLASS, 6.4))
    axes = np.atleast_2d(axes)
    row_of = {"positive": 0, "negative": 1}
    col_counter = {"positive": 0, "negative": 0}

    for _, r in samples.iterrows():
        img = load_normalized_image(r["image_path"])
        x = eval_transform(img).unsqueeze(0).to(DEVICE)
        cam, prob = grad_cam(model, x)
        base = np.array(img.resize((IMG_SIZE, IMG_SIZE)).convert("L"))

        row, col = row_of[r["stage1_label"]], col_counter[r["stage1_label"]]
        col_counter[r["stage1_label"]] += 1
        ax = axes[row, col]
        ax.imshow(base, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        pred = "positive" if prob >= 0.5 else "negative"
        ok = "OK" if pred == r["stage1_label"] else "WRONG"
        ax.set_title(f"true={r['stage1_label'][:3]} p(pos)={prob:.2f} {ok}", fontsize=8)
        ax.axis("off")

    for row in range(2):
        for col in range(col_counter[["positive", "negative"][row]], N_PER_CLASS):
            axes[row, col].axis("off")

    fig.suptitle(f"Grad-CAM (positive-class logit), random sample, {tag}", fontsize=10)
    plt.tight_layout()
    out = FIGS / f"gradcam_{tag}.png"
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

