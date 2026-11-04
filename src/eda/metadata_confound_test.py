import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

CSV = r"D:\EDI PROJECT 5\reports\eda\cov_caldas_dicom_metadata_analysis.csv"

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

# Convert categorical metadata to one-hot
X = pd.get_dummies(X, dummy_na=True)

# Numeric missing values
X = X.fillna(-999)

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
print(f"Features after encoding: {X.shape[1]}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

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

print("\nTop metadata features:")
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(20).to_string())
