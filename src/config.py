"""Central configuration for the pulmonary disease project."""

from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_pulmonary_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TARGET = "label"

FEATURES = [
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
]

CLASS_ORDER = [
    "healthy",
    "hypoxia_altitude",
    "hypoxia_anemic",
    "asthma",
    "edema",
    "fibrosis",
    "emphysema",
]
