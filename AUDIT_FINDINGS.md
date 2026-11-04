# CRITICAL FINDINGS: EDI PROJECT 5 - CNN COVID DETECTION AUDIT

## EXECUTIVE SUMMARY

**Status**: MULTIPLE CRITICAL FLAWS DETECTED  
**Verdict**: Current CNN results (AUC 1.0000) are NOT VALID for publication.  
**Root Cause**: Combination of acquisition artifacts, evaluation methodology issues, and age matching defects.

---

## 1. CRITICAL ISSUE: PERSISTENT ACQUISITION ARTIFACT AFTER NORMALIZATION

### Evidence
- CNN test AUC: 1.0000 (perfect)
- Technical feature RF AUC: 1.0000 (perfect) 
- Pixel-only RF AUC after normalization: 0.9961 (near-perfect)
- Metadata-only baseline AUC: 0.88-0.94 (significantly lower)

### Root Cause
Despite `load_normalized_image()` using percentile-based normalization, **images still retain strong class-specific intensity signatures**:

```
After normalization:
                negative  positive
mean              81.77    174.85   [2.1x difference]
border_mean       37.79    138.24   [3.7x difference]
center_mean      118.90    191.05   [1.6x difference]
frac_dark          0.39      0.05   [7.8x ratio]
frac_bright        0.02      0.05   [2.5x difference]
```

### Why normalization failed
The percentile-based normalization (clip to [1,99] percentile) assumes both classes have similar dynamic range distributions. They don't:
- **Positive images** (original uint16): Lower native intensity range → normalizes to BRIGHT after rescaling
- **Negative images** (original uint8): Already full range → normalizes to DARK/VARIABLE

This is an **acquisition artifact**, not medical signal. The two classes come from different sources/protocols with different exposure/processing.

### Impact
- Model has an easy shortcut: separate classes by intensity patterns alone
- **Age-matched test AUC 1.0000 is misleading** — the model isn't learning COVID, it's learning "is this a positive image by pixel intensity"

---

## 2. CRITICAL ISSUE: DEFECTIVE AGE-MATCHING PROCEDURE

### Val Set Residual Age Gap
The `build_age_matched_eval_set.py` script creates 10-year age bins and randomly selects equal numbers of patients from each bin. **This produces:**

```
VAL:
  Patients: 18 pos, 18 neg (balanced)
  AGE: positive mean 58.3, negative mean 70.3 → 12-YEAR GAP REMAINS
  
TEST:
  Patients: 22 pos, 22 neg (balanced)
  AGE: positive mean 58.3, negative mean 58.5 → 0.3-year gap (good)
```

### Root Cause
When bins have highly imbalanced positive/negative counts, random selection of equal numbers can result in very different individuals:
- Val [60,70): 14 positive patients, only 1 negative → randomly selects 1 of each, which may be 60 vs 70
- The matching algorithm doesn't minimize within-bin age distance; it just ensures equal class counts

### Image Imbalance
```
VAL matched:    36 patients, but 96 pos images vs 57 neg images (1.69x imbalance)
TEST matched:   44 patients, but 99 pos images vs 41 neg images (2.41x imbalance)
```
Because positive patients have more images (mean 4.25 in val), even patient-balanced matching produces image-imbalanced evaluation. **This upweights the positive class during image-level AUC calculation.**

### Impact
- Val age-matched results are contaminated by residual age confounding
- Image counts violate the matching assumption
- Small sample sizes (36-44 patients) make results statistically unreliable

---

## 3. CRITICAL ISSUE: AP/PA VIEW CONFOUND PERSISTS

### Distribution After Frontal Filtering
```
             AP (%)    PA (%)
Negative     70.5%     29.5%
Positive     87.1%     12.9%  ← SIGNIFICANTLY DIFFERENT
```

### Why This Matters
- AP (anterior-posterior) and PA (posterior-anterior) projections look visually different (depth cues, organ positioning)
- Positive class is **70% more likely to be AP-acquired** than negative
- This is a **test-time leakage**: if the model learns "AP → positive", it's not learning COVID

### Current Mitigation: None
The code acknowledges this ("Remaining confound: AP vs PA") but does NOT stratify evaluation or build balanced subsets.

### Impact
- Confound is documented but uncontrolled in final evaluation
- Removing AP/PA signal would lower CNN performance significantly

---

## 4. ISSUE: INVALID AGE-MATCHING METHODOLOGY

### What Was Intended
Create a patient subset where age doesn't predict label (test whether CNN learns beyond age shortcut).

### What Actually Happened
1. Patient-level age matching works (at test set)
2. Val set still has 12-year residual gap → **doesn't test what was intended**
3. Image-level evaluation with imbalanced images/patient ratio
4. Small sample size (18 positive patients in val) → **AUC noise floor ~0.15**

### Statistical Validity
```
Val matched: n=36 patients (18 per class)
  Min AUC uncertainty (worst case 50% random): ±0.15
  Report says: "AUC 1.0000 ± ?"
  Reality: This is likely a noisy maximum from a tiny sample
```

---

## 5. ISSUE: EVALUATION NOT AT INTENDED UNIT

### Train/Val/Test:   Per-patient weighting, but per-image evaluation
### Age-matched eval: Per-patient selection, but per-image evaluation with image count imbalance

This creates a mismatch: the model was trained to weight patients equally (via WeightedRandomSampler), but evaluation doesn't match that weighting assumption.

---

## SEVERITY RANKING

| Issue | Severity | Must Fix? |
|-------|----------|-----------|
| Artifact after normalization | CRITICAL | YES |
| Val age matching residual gap | CRITICAL | YES |
| Image count imbalance matched sets | HIGH | YES |
| AP/PA confound uncontrolled | HIGH | YES |
| Evaluation unit mismatch | MEDIUM | YES |
| Small sample size matched sets | MEDIUM | PARTIAL |

---

## WHAT TO TRUST FROM CURRENT RESULTS

✗ CNN test AUC 1.0000 → **INVALID, caused by artifact**
✗ CNN age-matched test AUC 1.0000 → **INVALID, caused by artifact + methodology issues**
✓ Metadata baseline AUC 0.88-0.94 → **VALID, this is the age/sex shortcut ceiling**
✓ Technical artifact AUC 1.0000 → **VALID AS EVIDENCE OF PROBLEM (not a good sign)**
? Grad-CAM visuals → **Cannot interpret until artifact is fixed**

---

## CONCLUSION

The CNN appears to be learning class-separable **pixel intensity patterns** from the acquisition artifact, not medical COVID-specific features. Even after filtering views and attempting age matching, the artifact persists because it's embedded in the image preprocessing and source characteristics.

**The model's "performance" is largely driven by:**
1. Intensity normalization differences (uint16 vs uint8 legacy)
2. Residual age confounding (especially in val set)
3. AP/PA acquisition direction differences
4. Possible genuine medical signal (cannot determine until artifacts are removed)

