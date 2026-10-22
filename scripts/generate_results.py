"""Regenerate the versioned figures and tables shown in the repository."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import FIGURES_DIR, SEED, TABLES_DIR
from data import load_or_generate_dataset
from evaluation import run_model_comparison, select_best_model
from models import get_model_specs
from preprocessing import split_dataset
from validation import validate_dataset
from visualization import (
    save_class_distribution,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_permutation_importance,
)


def main() -> None:
    dataset = load_or_generate_dataset(regenerate=True)
    validate_dataset(dataset)
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    comparison, fitted_models = run_model_comparison(
        X_train,
        X_test,
        y_train,
        y_test,
        get_model_specs(),
    )
    best_name, best_model = select_best_model(comparison, fitted_models)
    predictions = best_model.predict(X_test)

    cv_table = TABLES_DIR / "cv" / "model_comparison.csv"
    cv_table.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(cv_table, index=False)

    report = pd.DataFrame(
        classification_report(y_test, predictions, output_dict=True, zero_division=0)
    ).transpose()
    report_path = TABLES_DIR / "final_test" / "classification_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path)

    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": importance_result.importances_mean,
            "standard_deviation": importance_result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    importance_path = TABLES_DIR / "explainability" / "permutation_importance.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_path, index=False)

    save_class_distribution(dataset, FIGURES_DIR / "eda" / "class_distribution.png")
    save_correlation_heatmap(dataset, FIGURES_DIR / "eda" / "correlation_heatmap.png")
    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
        FIGURES_DIR / "final_test" / "confusion_matrix.png",
        f"Locked test set — {best_name}",
    )
    save_permutation_importance(
        importance.head(12),
        FIGURES_DIR / "explainability" / "permutation_importance.png",
    )

    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")


if __name__ == "__main__":
    main()
