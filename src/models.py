"""Model definitions and compact hyperparameter grids."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from config import SEED


def get_model_specs() -> dict:
    """Return the candidate classifiers used in the comparison."""
    return {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=SEED),
            "parameters": {
                "criterion": ["entropy"],
                "max_depth": list(range(1, 20)),
                "class_weight": [None, "balanced"],
            },
        },
        "Gaussian Naive Bayes": {
            "estimator": GaussianNB(),
            "parameters": {"var_smoothing": [10**exp for exp in range(-3, -13, -1)]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=SEED),
            "parameters": {
                "max_depth": list(range(4, 10)),
                "n_estimators": list(range(10, 60, 10)),
            },
        },
    }
