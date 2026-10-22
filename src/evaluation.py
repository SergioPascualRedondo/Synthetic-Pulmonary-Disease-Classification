"""Model selection and evaluation helpers."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def run_model_comparison(X_train, X_test, y_train, y_test, model_specs: dict, cv: int = 3):
    """Tune each candidate on macro-F1 and evaluate once on the locked test set."""
    rows = []
    fitted_models = {}

    for name, spec in model_specs.items():
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["parameters"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        fitted_models[name] = search
        rows.append(
            {
                "model": name,
                "cv_macro_f1": search.best_score_,
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "test_recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "test_macro_f1": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    return comparison, fitted_models


def select_best_model(comparison: pd.DataFrame, fitted_models: dict):
    """Select by development-set CV only, leaving test metrics descriptive."""
    best_name = comparison.iloc[0]["model"]
    return best_name, fitted_models[best_name]
