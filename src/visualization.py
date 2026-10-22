"""Publication-ready project visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from config import CLASS_ORDER


def save_class_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = dataset["label"].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set(title="Synthetic cohort class distribution", xlabel="Class", ylabel="Patients")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_correlation_heatmap(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(dataset.drop(columns="label").corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(model, X_test, y_test, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        labels=CLASS_ORDER,
        xticks_rotation=35,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_permutation_importance(importance: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered["feature"], ordered["importance"], color="#3569a8")
    ax.set(
        title="Permutation importance — selected model",
        xlabel="Decrease in macro-F1 after permutation",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
