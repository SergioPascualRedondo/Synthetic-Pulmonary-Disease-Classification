"""Leakage-safe dataset splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import SEED, TARGET


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = SEED,
):
    """Create a locked, stratified train/test split."""
    features = dataset.drop(columns=TARGET)
    target = dataset[TARGET]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
