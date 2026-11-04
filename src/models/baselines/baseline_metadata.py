"""
Phase 4 (Stage 1) — Metadata-only baseline: LogReg, RF, XGBoost on age + sex.

Purpose (two roles at once):
  1. The Phase 4 deliverable itself (ML baselines on metadata).
  2. The shortcut-learning sanity check for the CNN to come in Phase 5 — this
     is the same role a metadata-only baseline played in catching the
     original RICORD+ChestX-ray14 shortcut (CNN AUC ~0.9999 vs metadata AUC
     ~0.69 was the smoking gun). Given the confirmed 31.5-year age gap and
     17-point sex skew in Cov-Caldas, we EXPECT a real, non-trivial AUC here
     — the question for Phase 5 is whether the CNN's AUC is close to this
     ceiling (mostly shortcut) or well above it (learning real signal too).

Design decisions, stated explicitly rather than left implicit:
  - Trained and evaluated at the PATIENT level, not image level. age/sex are
    identical across all of a patient's images, and positive patients average
    4.88 images vs. negative's 2.62 (per the frozen-split decision log) — image-
    level duplication would let heavily-imaged patients dominate the loss and
    inflate AUC for reasons unrelated to age/sex signal quality.
  - age_is_estimated is EXCLUDED as a feature. It's a class-correlated
    artifact of the source data's session structure (18.6% negative vs 2.9%
    positive), not a clinical signal — including it would be training on a
    known confound, not measuring one.
  - Split (train/val/test) is the existing FROZEN patient-level split from
    Phase 2. No new splitting happens here.

Run from src/models/baselines/.
"""

from pathlib import Path
import argparse
import json
import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score,
                             f1_score, brier_score_loss)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).parent.parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "baselines"
REPORTS.mkdir(parents=True, exist_ok=True)
ARTIFACTS = ROOT / "models" / "metadata_baselines"

FEATURES = ["age", "sex_M"]  # age_is_estimated deliberately excluded — see module docstring


def load_patient_level():
    """Collapse the image-level master table to one row per patient.
    age/sex/label/split are constant within a patient (assumed, verified below)."""
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")

    # Verify the "constant within patient" assumption rather than assume it —
    # if it's ever violated, .first() below would silently pick an arbitrary
    # value instead of raising.
    n_inconsistent = 0
    for col in ["age", "sex", "stage1_label", "split"]:
        nunique_per_patient = df.groupby("patient_id")[col].nunique()
        bad = (nunique_per_patient > 1).sum()
        if bad:
            n_inconsistent += bad
            print(f"[FLAG] {bad} patients have inconsistent '{col}' across their images — inspect before trusting patient-level collapse.")
    assert n_inconsistent == 0, "Patient-level collapse assumption violated — fix before proceeding."

    patient_df = df.groupby("patient_id").agg({
        "age": "first",
        "sex": "first",
        "stage1_label": "first",
        "split": "first",
    }).reset_index()

    print(f"Collapsed {len(df)} images to {len(patient_df)} patients (1 row each).")
    return patient_df


def prepare_features(df):
    df = df.copy()
    sex = df["sex"].astype(str).str.strip().str.upper()
    df["sex_M"] = sex.map({"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0}).fillna(0.5)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("Metadata features contain missing ages; resolve them in the verification/preprocessing step.")
    df["label"] = (df["stage1_label"] == "positive").astype(int)
    return df


def evaluate(model, X, y, split_name):
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
    acc = accuracy_score(y, preds)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    pr_auc = average_precision_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
    brier = brier_score_loss(y, proba)
    print(f"  {split_name}: ROC-AUC={auc:.4f} PR-AUC={pr_auc:.4f} Acc={acc:.4f} F1={f1:.4f} Sens={sensitivity:.4f} Spec={specificity:.4f} (n={len(y)})")
    return {"split": split_name, "auc": auc, "pr_auc": pr_auc, "accuracy": acc,
            "precision": precision, "recall": recall, "f1": f1,
            "sensitivity": sensitivity, "specificity": specificity,
            "brier_score": brier, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": len(y)}


def run_model(name, model, train, val, test, save=True):
    print(f"\n--- {name} ---")
    X_train, y_train = train[FEATURES], train["label"]
    if y_train.nunique() != 2:
        raise ValueError(f"Training split has one class only; cannot fit {name}.")
    model.fit(X_train, y_train)

    # Save each estimator separately so it can be loaded without retraining.
    if save:
        model_dir = ARTIFACTS / name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model": name, "features": FEATURES,
                       "target": "stage1_label == positive",
                       "unit": "patient", "seed": 42,
                       "sex_M_mapping": {"male": 1, "female": 0, "unknown": 0.5},
                       "hyperparameters": model.get_params(deep=False)}, f, indent=2, default=str)

    results = []
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        proba = model.predict_proba(frame[FEATURES])[:, 1]
        results.append(evaluate(model, frame[FEATURES], frame["label"], split_name))
        pd.DataFrame({"patient_id": frame["patient_id"].to_numpy(),
                      "label": frame["label"].to_numpy(),
                      "probability_positive": proba}).to_csv(
                          model_dir / f"{split_name}_predictions.csv", index=False)

    if hasattr(model, "coef_"):
        print(f"  Coefficients: age={model.coef_[0][0]:.4f}  sex_M={model.coef_[0][1]:.4f}")
        print("  Coefficients are descriptive associations in this training split, not causal effects.")
    elif hasattr(model, "feature_importances_"):
        print(f"  Feature importances: age={model.feature_importances_[0]:.4f}  sex_M={model.feature_importances_[1]:.4f}")

    for r in results:
        r["model"] = name
    if save:
        pd.DataFrame(results).to_csv(model_dir / "metrics.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Cov-Caldas Stage 1 metadata baselines.")
    parser.add_argument("--model", choices=["all", "logistic", "random_forest", "xgboost"],
                        default="all", help="Train one baseline independently, or all available models.")
    args = parser.parse_args()
    patient_df = load_patient_level()
    patient_df = prepare_features(patient_df)

    train = patient_df[patient_df["split"] == "train"]
    val = patient_df[patient_df["split"] == "val"]
    test = patient_df[patient_df["split"] == "test"]
    print(f"\nPatients — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Positive rate — train: {train['label'].mean():.3f}, val: {val['label'].mean():.3f}, test: {test['label'].mean():.3f}")

    all_results = []

    if args.model in ("all", "logistic"):
        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        all_results += run_model("LogisticRegression", logreg, train, val, test)

    if args.model in ("all", "random_forest"):
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, max_depth=5)
        all_results += run_model("RandomForest", rf, train, val, test)

    if args.model in ("all", "xgboost") and HAS_XGB:
        pos_weight = (train["label"] == 0).sum() / (train["label"] == 1).sum()
        xgb = XGBClassifier(n_estimators=200, max_depth=3, scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=42)
        all_results += run_model("XGBoost", xgb, train, val, test)
    elif args.model == "xgboost" and not HAS_XGB:
        raise RuntimeError("XGBoost requested but not installed. Install it with: python -m pip install xgboost")
    elif args.model == "all" and not HAS_XGB:
        print("\n[SKIPPED] xgboost not installed — run `pip install xgboost` to include it. "
              "LogReg + RF results below are unaffected.")

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        raise RuntimeError("No model was trained.")
    if args.model == "all":
        results_df.to_csv(REPORTS / "cov_caldas_metadata_baseline_results.csv", index=False)

    print(f"\n=== Summary (test set AUC — this is the shortcut-learning ceiling for Phase 5) ===")
    test_summary = results_df[results_df["split"] == "test"][["model", "auc", "accuracy", "sensitivity", "specificity"]]
    print(test_summary.to_string(index=False))
    print(f"\nIf the Phase 5 CNN's test AUC is close to the best of these numbers, that's evidence")
    print(f"it's leaning on age/sex-correlated image features (shortcut learning) rather than")
    print(f"genuine COVID-specific signal — same interpretation that caught the original pivot's problem.")

    if args.model == "all":
        print(f"\nSaved combined results to {REPORTS / 'cov_caldas_metadata_baseline_results.csv'}")
    else:
        print(f"\nSaved standalone model, config, and metrics under {ARTIFACTS}")


if __name__ == "__main__":
    main()
