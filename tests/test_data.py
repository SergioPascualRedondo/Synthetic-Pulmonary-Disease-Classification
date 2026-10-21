from __future__ import annotations

import pandas as pd

from pulmonary_disease.data import generate_pulmonary_synthetic_dataset

EXPECTED_COLUMNS = [
    "Age",
    "HR",
    "RR",
    "SpO2",
    "PaCO2",
    "HCO3",
    "pH",
    "Hb",
    "CaO2",
    "AirwayResistance",
    "FEV1_FVC",
    "DLCO_percent",
    "B_lines",
    "CXR_score",
    "BNP",
    "label",
]

EXPECTED_LABELS = {
    "healthy",
    "hypoxia_altitude",
    "hypoxia_anemic",
    "asthma",
    "edema",
    "fibrosis",
    "emphysema",
}


def test_default_dataset_schema_and_size() -> None:
    dataset = generate_pulmonary_synthetic_dataset(seed=42)
    assert dataset.shape == (2150, 16)
    assert dataset.columns.tolist() == EXPECTED_COLUMNS
    assert set(dataset["label"].unique()) == EXPECTED_LABELS
    assert not dataset.isna().any().any()


def test_generation_is_deterministic() -> None:
    first = generate_pulmonary_synthetic_dataset(seed=42)
    second = generate_pulmonary_synthetic_dataset(seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_physiological_bounds() -> None:
    dataset = generate_pulmonary_synthetic_dataset(seed=42)
    bounds = {
        "Age": (18, 90),
        "HR": (45, 160),
        "RR": (10, 40),
        "SpO2": (55, 100),
        "PaCO2": (24, 75),
        "HCO3": (14, 40),
        "pH": (7.02, 7.58),
        "Hb": (6.5, 18.5),
        "CaO2": (4.5, 24.5),
        "AirwayResistance": (0.8, 10.0),
        "FEV1_FVC": (0.30, 0.92),
        "DLCO_percent": (18, 125),
        "B_lines": (0, 30),
        "CXR_score": (0, 12),
        "BNP": (5, 3500),
    }
    for column, (minimum, maximum) in bounds.items():
        assert dataset[column].between(minimum, maximum).all(), column

