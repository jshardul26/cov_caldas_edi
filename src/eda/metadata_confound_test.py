import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "reports" / "eda" / "cov_caldas_dicom_metadata_analysis.csv"

df = pd.read_csv(CSV)

print("=" * 80)
print("METADATA-ONLY ACQUISITION CONFOUND TEST")
print("=" * 80)

# Only metadata known before looking at image pixels
FEATURES = [
    "manufacturer",
    "model",
    "software",
    "view_position",
    "bits_allocated",
    "bits_stored",
    "high_bit",
    "photometric",
    "pixel_representation",
    "window_center",
    "window_width",
    "rescale_intercept",
    "rescale_slope",
    "rescale_type",
    "presentation_lut",
]

X = df[FEATURES].copy()
y = (df["stage1_label"] == "positive").astype(int)

train_mask = df["split"] == "train"
val_mask = df["split"] == "val"
test_mask = df["split"] == "test"

X_train = X.loc[train_mask]
X_val = X.loc[val_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_val = y.loc[val_mask]
y_test = y.loc[test_mask]

print(f"\nTrain: {len(X_train)}")
print(f"Val:   {len(X_val)}")
print(f"Test:  {len(X_test)}")
print(f"Raw metadata features: {X.shape[1]}")

categorical = [c for c in FEATURES if X[c].dtype == "object"]
numeric = [c for c in FEATURES if c not in categorical]
preprocess = ColumnTransformer([
    ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
    ("numeric", SimpleImputer(strategy="median"), numeric),
])
rf = make_pipeline(preprocess, RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
))

rf.fit(X_train, y_train)

val_prob = rf.predict_proba(X_val)[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]

val_auc = roc_auc_score(y_val, val_prob)
test_auc = roc_auc_score(y_test, test_prob)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"Metadata-only validation AUC: {val_auc:.4f}")
print(f"Metadata-only test AUC:       {test_auc:.4f}")

print("\nInterpretation:")
print("AUC near 0.50 -> metadata does not separate classes well.")
print("AUC substantially above 0.50 -> acquisition metadata contains class information.")
print("Very high AUC -> strong acquisition/domain confounding.")

print("\nPreprocessing note:")
print("The fitted pipeline learns categorical encodings from training data only.")
