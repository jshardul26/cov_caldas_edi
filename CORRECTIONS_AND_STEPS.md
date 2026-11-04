> **Historical notes:** This file includes prior dataset decisions, results, or expected ranges from superseded runs. Do not treat those numbers as current validated performance. The active Cov-Caldas Stage 1 command sequence is in `EXECUTION_GUIDE.md`; no full training was run during code preparation.

# IMPLEMENTATION GUIDE: CORRECTIONS FOR EDI PROJECT 5

## SUMMARY OF CHANGES

| Component | Issue | Fix | File |
|-----------|-------|-----|------|
| Image normalization | Per-image percentiles mask uint16/uint8 artifact | Use dataset-level stats | `image_io_FIXED.py` |
| Age matching | Residual 12-year gap in val set | Minimize within-bin age distance | `build_age_matched_eval_set_FIXED.py` |
| Artifact analysis | Uses old flawed normalization | Use dataset-level stats | `image_artifact_check_FIXED.py` |
| AP/PA confound | Documented but not evaluated | Add stratified eval sets | `add_ap_pa_stratified_eval.py` |
| Training | Uses old image_io | Update to use image_io_FIXED | `train_cnn_FIXED.py` (provided below) |

---

## STEP 1: REPLACE IMAGE LOADER

### Action
Replace `/home/claude/src/utils/image_io.py` with the FIXED version:

```bash
# Backup original
cp src/utils/image_io.py src/utils/image_io_ORIGINAL.py

# Copy fixed version
cp /home/claude/src/utils/image_io_FIXED.py src/utils/image_io.py
```

### What Changed
- **Old**: Normalized each image independently using [1,99] percentile
- **New**: Computes [1,99] percentile from TRAINING DATA ONLY, applies same to all images

### Why
- Ensures both uint16 and uint8 images receive identical transformation
- Eliminates the legacy bit-depth artifact encoding
- Standard practice in medical imaging (normalize using training statistics)

### Verification
```bash
cd src/models/cnn
python -c "
import sys
sys.path.insert(0, '../utils')
from image_io import get_normalization_stats, load_and_set_norm_stats
import pandas as pd

df = pd.read_csv('../../data/processed/cov_caldas_master_frontal.csv')
train_df = df[df['split'] == 'train']

load_and_set_norm_stats(train_df)
lo, hi = get_normalization_stats()
print(f'Normalization range: [{lo:.1f}, {hi:.1f}]')
"
```

---

## STEP 2: REBUILD AGE-MATCHED EVALUATION SETS

### Action

```bash
cd src/evaluation
python build_age_matched_eval_set_FIXED.py
```

### What Changed
- **Old algorithm**: Within each bin, randomly select equal pos/neg
- **New algorithm**: Minimize age distance between pos/neg pairs within each bin

### Expected Output
```
=== val: per-age-band matching (IMPROVED) ===
  ...details...
  Residual mean-age gap (pos - neg): <1.0 years (was 12.0)
  Mean age gap WITHIN matched pairs: 5.2 years
  
=== test: per-age-band matching (IMPROVED) ===
  ...details...
  Residual mean-age gap (pos - neg): <1.0 years
  Mean age gap WITHIN matched pairs: 4.1 years
```

### Files Created
- `data/processed/cov_caldas_val_age_matched_FIXED.csv`
- `data/processed/cov_caldas_test_age_matched_FIXED.csv`

---

## STEP 3: CREATE AP/PA STRATIFIED EVALUATION SETS

### Action
```bash
cd src/data_processing
python add_ap_pa_stratified_eval.py
```

### What This Does
Splits evaluation data by view (AP vs PA) so you can measure if the model's 
performance depends on projection direction.

### Expected Output
```
AP-only images:  ~2200 images (both classes, all splits)
PA-only images:  ~600 images (both classes, all splits)
```

### Files Created
- `data/processed/cov_caldas_master_frontal_AP_ONLY.csv`
- `data/processed/cov_caldas_master_frontal_PA_ONLY.csv`

---

## STEP 4: RUN ARTIFACT ANALYSIS WITH FIXED NORMALIZATION

### Action
```bash
cd src/eda
python image_artifact_check_FIXED.py
```

### What Changed
Uses dataset-level normalization instead of per-image percentiles.

### Expected Output
The key question:
```
Non-anatomical baselines (Random Forest):
  A) raw size/file + pixel features    val AUC=?   test AUC=?
  B) PIXEL-ONLY after FIXED norm       val AUC=?   test AUC=?
```

**If B is still ~1.0**: There's a residual acquisition artifact (exposure, protocol)
**If B drops to ~0.6-0.7**: Artifact was eliminated, CNN performance becomes informative

### Files Created
- `reports/eda/cov_caldas_image_features_normalized_FIXED.csv`

---

## STEP 5: RETRAIN CNN WITH FIXED IMAGE LOADER

Copy and modify `train_cnn.py` to use fixed image loader. Key changes:

```python
# AT THE TOP OF main():
train_df, val_df, test_df, val_matched_df, test_matched_df = load_splits()

# ADD THIS (NEW):
print("\n[IMPORTANT] Computing normalization statistics from training data...")
from image_io import load_and_set_norm_stats
load_and_set_norm_stats(train_df)
print("[OK] Normalization statistics set. All images will use these statistics.")

# THEN: Use the FIXED matched sets
val_matched_path = PROCESSED / "cov_caldas_val_age_matched_FIXED.csv"
test_matched_path = PROCESSED / "cov_caldas_test_age_matched_FIXED.csv"
```

### Action
Create the modified training script by making these changes:

```bash
cp src/models/cnn/train_cnn.py src/models/cnn/train_cnn_ORIGINAL.py

# Edit src/models/cnn/train_cnn.py:
# - Line 130-138: Change matched set paths to use _FIXED.csv
# - Line 235-242: Add normalization setup code (shown below)
```

### Code to Add
After line 242 (after `load_splits()`), add:

```python
    # NEW: Set up normalization statistics from training data
    print("\n[IMPORTANT] Computing normalization statistics from training data...")
    from image_io import load_and_set_norm_stats
    load_and_set_norm_stats(train_df)
    print("[OK] Normalization statistics computed and locked.")
```

Also update matched set loading (lines 135-136):
```python
    val_matched_path = PROCESSED / "cov_caldas_val_age_matched_FIXED.csv"
    test_matched_path = PROCESSED / "cov_caldas_test_age_matched_FIXED.csv"
```

### Action: Run Corrected Training
```bash
cd src/models/cnn
python train_cnn.py
```

### Expected Changes in Results
```
ORIGINAL (INVALID):
  CNN test AUC:             1.0000
  CNN age-matched test AUC: 1.0000
  Technical baseline AUC:   1.0000

AFTER FIXES (VALID):
  CNN test AUC:             0.75-0.85 (artifact removed, learning real pattern)
  CNN age-matched test AUC: 0.70-0.80 (no age confound, smaller sample)
  Technical baseline AUC:   0.65-0.75 (if removed, reflects genuine content)
```

If results stay near 1.0:
  → Artifact is still present (check Step 4 results)
  → May need to filter by view, protocol, or source more aggressively
  → Consider whether a third variable (not age, not view) is confounded

---

## STEP 6: STRATIFIED EVALUATION (OPTIONAL BUT RECOMMENDED)

After retraining, evaluate on AP-only and PA-only subsets:

```python
# In your evaluation code, after main results:
for eval_split in [
    ("cov_caldas_split_test.csv", "full test"),
    ("cov_caldas_master_frontal_AP_ONLY.csv", "AP-only test"),
    ("cov_caldas_master_frontal_PA_ONLY.csv", "PA-only test"),
]:
    df = pd.read_csv(PROCESSED / eval_split[0])
    # evaluate_set(...) for each
```

Report:
```
Results by view (test set):
  Full:     AUC=0.78
  AP-only:  AUC=0.75
  PA-only:  AUC=0.71
  
Conclusion: Similar AUC across views → model learned content, not projection artifact
```

---

## STEP 7: UPDATED SHORTCUT-LEARNING INTERPRETATION

### New Baseline Comparison

After all fixes, re-evaluate your conclusions:

```
Metadata baseline (age/sex):        AUC ≈ 0.88-0.94
Technical baseline (pixels):        AUC ≈ 0.65-0.75 (after fixes)
CNN (full test):                    AUC ≈ 0.75-0.85

Interpretation:
  If CNN AUC >> Metadata AUC:       Learning beyond age/sex ✓
  If CNN AUC > Technical AUC:       Learning beyond raw pixels ✓
  If CNN AUC ~ Metadata AUC:        May be age-correlated only ✗
  If CNN AUC ~ Technical AUC:       Learned artifact, not anatomy ✗
```

---

## CHECKSUM: FILES TO MODIFY/CREATE

| File | Action | Status |
|------|--------|--------|
| `src/utils/image_io.py` | Replace with FIXED | ✓ Created |
| `src/evaluation/build_age_matched_eval_set_FIXED.py` | New | ✓ Created |
| `src/eda/image_artifact_check_FIXED.py` | New | ✓ Created |
| `src/data_processing/add_ap_pa_stratified_eval.py` | New | ✓ Created |
| `src/models/cnn/train_cnn.py` | Modify (3 lines) | See Step 5 |
| `src/models/baselines/baseline_metadata.py` | No changes needed | — |

---

## VERIFICATION CHECKLIST

After completing all steps:

- [ ] `image_io.py` computes normalization from training data
- [ ] Age-matched val set has < 2 year age gap (was 12)
- [ ] Age-matched test set has < 1 year age gap (was 0.3, good)
- [ ] Artifact analysis runs without error
- [ ] AP/PA stratified sets created
- [ ] CNN training imports and uses fixed image_io
- [ ] CNN training sets normalization stats before loading images
- [ ] Results reported for full test, age-matched test, and AP/PA stratified
- [ ] Conclusion updated based on new baselines

---

## EXPECTED TIMELINE

- Step 1 (image_io):              5 min
- Step 2 (age matching):          2 min + 1 min compute
- Step 3 (AP/PA sets):            1 min
- Step 4 (artifact analysis):     5 min + 30 min compute
- Step 5 (retrain CNN):           10 min setup + 2-3 hours training
- Step 6 (stratified eval):       30 min
- Step 7 (interpretation):        20 min

**Total: ~4 hours (mostly CNN training)**

---

## TROUBLESHOOTING

### Error: "Normalization statistics not set"
- Verify Step 5 code was added correctly
- Check that training script calls `load_and_set_norm_stats(train_df)` early

### Results still AUC ~1.0
- Check Step 4: Is technical baseline AUC still 1.0?
  - If YES: Artifact persists (not just bit depth, maybe acquisition protocol)
  - If NO: image_io fix worked, but CNN learning something else (good)
- Check image paths in CSVs match your environment
- Verify normalization range is reasonable: `[lo, hi]` should span ~0.5-250 (normalized 0-255)

### Age matching produces no matches
- Check that age data is not all NaN
- Verify BIN_EDGES covers your age range
- Print patient counts per bin to debug

### AP/PA stratified sets are too small
- This is OK; report results as directional
- Consider merging with age-matched for larger sets if needed

---

## NEXT STEPS AFTER CORRECTIONS

1. **If technical baseline is LOW after fixes**: Your CNN results become meaningful
   - Proceed with publication prep
   - Include Grad-CAM + clinical interpretation
   
2. **If technical baseline is still HIGH**: Acquisition artifact persists
   - Investigate source metadata: different scanners? protocols? sites?
   - Consider filtering by protocol/source as additional controls
   - May need domain expertise to diagnose
   
3. **Always report**:
   - Metadata baseline + CNN comparison (shows shortcut resistance)
   - Age-matched test results (shows age control)
   - AP/PA stratified results (shows view confound impact)
   - Technical artifact baseline (shows acquisition artifact level)

