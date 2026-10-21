"""Execute the analysis notebook in a fresh kernel and save the rendered report."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "pulmonary_disease_classification.ipynb"
DEFAULT_OUTPUT = ROOT / "reports" / "pulmonary_disease_classification_executed.ipynb"


def execute_notebook(source: Path, output: Path, timeout: int) -> Path:
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum seconds allowed for each notebook cell (default: 1800).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = execute_notebook(SOURCE, args.output.resolve(), args.timeout)
    print(f"Executed notebook written to: {result}")

