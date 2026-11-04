"""Small dataset-free checks; these never launch a training loop."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "data_processing"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from train_stage1_dl import FusionNet, MetadataMLP
from build_cov_caldas_split import frozen_patient_split
from build_age_matched_eval_set_FIXED import match_split_improved


def test_patient_split_keeps_patients_disjoint():
    df = pd.DataFrame([
        {"patient_id": f"{label}_{i}", "stage1_label": label, "image_path": f"{label}_{i}.png"}
        for label in ("positive", "negative") for i in range(20)
    ])
    result = frozen_patient_split(df, seed=42)
    ids = [set(result.loc[result.split == split, "patient_id"]) for split in ("train", "val", "test")]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))


def test_age_match_pairs_unique_closest_patient():
    rows = [
        {"patient_id": pid, "age": age, "stage1_label": label, "image_path": f"{pid}.png", "split": "val"}
        for pid, age, label in (("p1", 41, "positive"), ("p2", 43, "positive"),
                                ("n1", 42, "negative"), ("n2", 80, "negative"))
    ]
    matched = match_split_improved(pd.DataFrame(rows), "synthetic")
    assert set(matched.patient_id) == {"p1", "n1"}


def test_metadata_and_fusion_forward_and_reload(tmp_path):
    metadata = torch.randn(2, 2)
    mlp = MetadataMLP().eval()
    assert mlp(metadata).shape == (2,)
    torch.save(mlp.state_dict(), tmp_path / "mlp.pt")
    restored = MetadataMLP().eval()
    restored.load_state_dict(torch.load(tmp_path / "mlp.pt", map_location="cpu", weights_only=True))
    assert torch.allclose(mlp(metadata), restored(metadata))

    fusion = FusionNet(pretrained=False).eval()
    with torch.no_grad():
        logits = fusion(torch.randn(1, 3, 64, 64), metadata[:1])
    assert logits.shape == (1,)
