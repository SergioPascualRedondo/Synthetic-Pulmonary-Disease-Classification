# Reproducibility protocol

## Clean execution

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Expected artifacts

- `data/processed/synthetic_pulmonary_dataset.csv`
- `reports/pulmonary_disease_classification_executed.ipynb`

The notebook regenerates the CSV rather than relying on the committed copy. The committed dataset allows immediate inspection and provides a baseline artifact.

## Verification

Run:

```bash
pytest
```

The tests confirm deterministic generation, schema, cohort size, class coverage, absence of missing values, and declared physiological bounds.

## Provenance

`reports/baseline_executed_notebook.ipynb` preserves the original executed notebook used to initialize this repository. The clean notebook under `notebooks/` fixes the original hidden-kernel-state dependency and can run from a fresh Python process.

