# Stage 1 research paper draft — Cov-Caldas COVID-19 detection

> Draft scaffold only. Complete from verified provenance and measured experiments. Bracketed fields are placeholders; no results are asserted or fabricated. Stage 2 severity is outside this draft.

## Title

Patient-level evaluation of image-only, metadata-only, and multimodal models for COVID-19 detection in chest radiographs from the Cov-Caldas dataset

## Abstract

**Background:** [Problem and clinical/research context.]
**Objective:** Compare image-only, age/sex-only, and fused image plus age/sex models for binary COVID-19 classification using Cov-Caldas.
**Methods:** [Dataset provenance, cohort, patient-level split, preprocessing, models, metrics.]
**Results:** [Insert actual held-out test results only after evaluation. Include uncertainty intervals.]
**Conclusion:** [Conclusion limited to evidence and dataset scope.]

## 1. Introduction

[Motivation, chest radiography role, risks of shortcut learning, and study objectives.]

## 2. Related work

[Cite verified peer-reviewed image-only and multimodal CXR studies; compare dataset and evaluation design.]

## 3. Research questions

1. How does the image-only ResNet-18 compare with age/sex-only models?
2. Does age/sex fusion change held-out patient performance relative to the image-only model?
3. How stable are results across AP and PA views and age-matched evaluation subsets?

## 4. Dataset and cohort

[Cite Cov-Caldas paper and dataset landing page. Report only counts reproduced by `verify_cov_caldas.py`; document the public release subset and inclusion/exclusion criteria.]

## 5. Ethics and data use

[Document source consent/approval, de-identification, license restrictions, and institutional requirements.]

## 6. Labels and target

Binary `positive` versus `negative` target as supplied by Cov-Caldas class directories. Document any label uncertainty and avoid claims about clinical diagnosis.

## 7. Patient-level preprocessing

[Metadata cleaning, ambiguous age handling, image formats, bit-depth harmonization, normalization, frontal filtering.]

## 8. Data partitioning

[Describe frozen patient-level stratified train/validation/test assignments, seed, counts, and a patient overlap check.]

## 9. Models

Describe Logistic Regression, Random Forest, optional XGBoost, metadata MLP, image-only ResNet-18, and multimodal ResNet-18 plus age/sex fusion. Specify which metadata fields each model receives.

## 10. Training procedure

[Hyperparameters from saved config files, class weighting, patient-level sampling, selection by validation ROC-AUC, early stopping, hardware/software.]

## 11. Evaluation metrics

Report ROC-AUC, PR-AUC, accuracy, precision, recall/sensitivity, specificity, F1, Brier score, and confusion matrix. Include patient-level results where the evaluation unit is patient-level and state the unit for every table.

## 12. Statistical analysis

[Confidence intervals and comparison approach chosen before interpreting test results.]

## 13. Main results

[Populate from held-out evaluation files after all model runs. Do not use training results as evidence of generalization.]

## 14. Ablation and subgroup analyses

[Compare metadata-only, image-only, and fusion; report AP/PA and age-matched subsets with sample counts and uncertainty.]

## 15. Explainability

[Show randomly selected Grad-CAM examples and metadata SHAP summaries. Explain these as qualitative tools, not proof of causal or clinical reasoning.]

## 16. Error analysis

[Review false positives/negatives using a prespecified process; report subgroup counts and limitations.]

## 17. Discussion

[Interpret only observed, uncertainty-aware findings relative to the research questions.]

## 18. Limitations

[Single-source, public-release size, class imbalance, demographic coverage, image acquisition variation, sample size, and non-clinical evaluation limitations.]

## 19. Conclusion

[Complete after results are available.]

## 20. References

[Add verified bibliographic records and dataset citations. Do not cite unverified generated claims.]
