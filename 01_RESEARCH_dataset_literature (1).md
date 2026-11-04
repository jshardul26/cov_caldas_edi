# Phase 0 — Dataset & Literature Research (FINALIZED)

Status: **FINAL — LOCKED.** COVID-19, two-stage architecture, two datasets locked, no further branching (see `00_PROJECT_MASTER.md` §4 for the full decision trail including the BIMCV reconsideration and rejection on disk-space grounds). The earlier melanoma pivot proposal (Sept 26 draft) is superseded and kept only in the Decisions Log for audit trail.

---

## I. Final dataset decision

### Dataset 1 — MIDRC-RICORD-1c (positive class + severity labels)

| Field | Value |
|---|---|
| Source | The Cancer Imaging Archive (TCIA), collab RSNA + Society of Thoracic Radiology |
| Content | 998 chest X-ray DICOM images (DX, CR) from 361 COVID-positive patients, 4 international sites |
| Size | 12.27 GB |
| License | CC BY-NC 4.0 |
| Access | Immediate — no registration/approval wait. Requires TCIA/NBIA Data Retriever tool for the image manifest. |
| Clinical metadata (XLSX) | MRN*, Age, Study Date*, Exam Description, Sex, Study UID*, Image Count, Modality, Testing Result, Specimen Source (* = pseudonymous) |
| Labels (JSON annotations) | Classification: Typical / Indeterminate / Atypical / Negative for Pneumonia. Airspace Disease Grading: Mild (1-2 lung zones) / Moderate (3-4 zones) / Severe (>4 zones) |
| Join key | MRN (patient-level) + Study UID (study-level) — present in both the clinical XLSX and the DICOM header tags (PatientID, StudyInstanceUID) |
| Role in project | Positive class for Stage 1 (detection); full dataset for Stage 2 (severity) |

### Dataset 2 — NIH ChestX-ray14 (negative class)

| Field | Value |
|---|---|
| Source | NIH Clinical Center |
| Content | Filter full 112,120-image / 30,805-patient dataset down to the **"No Finding"** label; randomly sample ~1,000 images to roughly match RICORD-1c's 998 positives |
| License | No usage restrictions; attribution required (NIH download link + CVPR 2017 citation) |
| Access | Immediate — mirrored on Kaggle and Hugging Face (`alkzar90/NIH-Chest-X-ray-dataset`), no registration |
| Metadata | `Data_Entry_2017_v2020.csv`: Patient ID, Image Index (filename), Age, Sex, View Position (AP/PA) |
| Join key | `Patient ID` + `Image Index` — direct CSV-to-filename merge |
| Role in project | Negative class for Stage 1 (detection) only |

### Why this pair over the alternatives already audited

- **COVID-19-NY-SBU** (1,384 patients, richest metadata, real mortality outcome) — kept as a documented stretch/enrichment option, not primary: 511.5 GB unfiltered, and even CR/DX-filtered it needs OMOP-CDM cleaning work the 2-day-then-24-hour runway can't safely absorb first.
- **COVID-19-AR** (105 patients, comorbidities + genomics) — too small alone; noted as a possible small external-validation set later, not now.
- **POLCOVID** (4,809 images, 15 hospitals, smoking status in metadata) — rejected for now: registration-gated access with unverified turnaround time.
- **BIMCV-COVID19+ / BIMCV-COVID19-** — genuinely reconsidered on 2026-09-27, since it's same-source (would remove the domain-shift issue below entirely) and its structured UMLS-mapped labels turned out not to require Spanish-language processing as first assumed. **Rejected anyway: team's laptop has under 50 GB free disk space**, and BIMCV's real download size was never confirmed to fit that — not worth the risk this close to deadline. **This is now closed, not a live option.**
- **MIMIC-CXR** — rejected: weeks-long PhysioNet credentialing, not feasible under any deadline discussed.

### Known, honest limitation (carry into paper's Limitations section)

RICORD-1c (4 international sites) and ChestX-ray14 (NIH Clinical Center, US) are from different hospitals/scanners/populations. Stage 1 (detection) is therefore a **cross-dataset task with a real domain-shift risk** — the model could learn "which source dataset" rather than "COVID or not." Planned mitigation/check: train a metadata-only classifier (age/sex/view) on the Stage 1 labels; if it separates the two classes suspiciously well on its own, that confirms the shift, and it's reported plainly rather than hidden. Stage 2 (severity) does not have this problem — it uses RICORD-1c exclusively.

---

## II. Two-stage architecture

```text
STAGE 1 — Detection (COVID+ vs. COVID-)
  Images: RICORD-1c positives + ChestX-ray14 "No Finding" negatives
  Metadata: shared minimal feature set — Age, Sex (the only fields both datasets have)
                     |
        CNN branch (image embedding)      MLP branch (metadata embedding)
                     \                           /
                      -->  Fusion  -->  P(COVID+)

STAGE 2 — Severity (positives only, RICORD-1c)
  Images: RICORD-1c only
  Metadata: full RICORD-1c clinical fields (Age, Sex, Testing Result, Specimen Source)
                     |
        CNN branch (image embedding)      MLP branch (metadata embedding)
                     \                           /
                      -->  Fusion  -->  Mild / Moderate / Severe
```

Both stages can share the same pretrained CNN backbone (fine-tune separate heads) to save training time given the RTX 4050 budget.

---

## III. Literature gap (unchanged from Sept 26 audit, still applies)

Prior multimodal COVID CXR work (e.g. the AI4COVID hackathon dataset study, BIO-CXRNET) has shown metadata fusion helps for severity/mortality prediction, but typically as a single task on a single dataset. Our incremental, defensible gap: a **two-stage pipeline** (detection → severity) evaluated with the same rigor (full ablation: image-only vs. metadata-only vs. fused, per stage) plus an explicit, measured domain-shift check on the detection stage — a combination not found together in the papers surveyed Sept 26.

---

## IV. Revised experimental plan

1. Metadata → Logistic Regression, Random Forest, XGBoost (Stage 1, shared age/sex features)
2. Image → CNN transfer learning (shared backbone, Stage 1 head)
3. Metadata → MLP (Stage 1)
4. Image + Metadata → Fusion (Stage 1) → ablation → domain-shift check
5. Metadata → LogReg/RF/XGBoost + MLP (Stage 2, full RICORD fields, positives only)
6. Image → CNN (Stage 2 head, fine-tuned from Stage 1 backbone)
7. Image + Metadata → Fusion (Stage 2) → ablation
8. Explainability: Grad-CAM (both stages' image branches) + SHAP (both stages' metadata branches)
9. *(Stretch, post-Monday)* COVID-19-AR or COVID-19-NY-SBU (CR/DX-filtered) as external validation

## V. Roadmap (24-hour execution plan agreed)

See conversation log / team chat for the hour-by-hour breakdown. Checkpoint priority if time runs short: **ship a complete, honestly-evaluated Stage 1 before starting Stage 2** — a working single stage beats a rushed, unverified second stage.
