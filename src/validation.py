"""Dataset integrity checks used by analysis scripts and tests."""

import pandas as pd

from config import CLASS_ORDER, FEATURES, TARGET


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Raise a clear error when the generated dataset violates its contract."""
    expected_columns = FEATURES + [TARGET]
    if list(dataset.columns) != expected_columns:
        raise ValueError("Unexpected dataset schema")
    if dataset.empty:
        raise ValueError("The dataset is empty")
    if dataset.isna().any().any():
        raise ValueError("The dataset contains missing values")
    if set(dataset[TARGET].unique()) != set(CLASS_ORDER):
        raise ValueError("Unexpected target classes")
