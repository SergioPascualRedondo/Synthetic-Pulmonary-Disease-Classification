"""Dataset access helpers."""

from pathlib import Path

import pandas as pd

from config import DATA_PATH, SEED
from pulmonary_disease.data import generate_pulmonary_synthetic_dataset


def load_or_generate_dataset(
    path: Path = DATA_PATH,
    seed: int = SEED,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load the versioned dataset or regenerate it deterministically."""
    if path.exists() and not regenerate:
        return pd.read_csv(path)

    dataset = generate_pulmonary_synthetic_dataset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return dataset
