# Reproducibility protocol

## Clean execution

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/generate_results.py
python scripts/reproduce.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Expected artifacts

- `data/raw/synthetic_pulmonary_dataset.csv`
- `results/figures/eda/`
- `results/figures/explainability/`
- `results/figures/final_test/`
- `results/tables/cv/`
- `results/tables/explainability/`
- `results/tables/final_test/`
- `reports/pulmonary_disease_classification_executed.ipynb`

The results script regenerates the CSV, model comparison, final evaluation, and interpretation artifacts rather than relying on hidden notebook state. The committed files allow immediate inspection and provide a reproducible baseline.

## Verification

Run:

```bash
pytest
```

The tests confirm deterministic generation, schema, cohort size, class coverage, absence of missing values, and declared physiological bounds.

## Provenance

`reports/baseline_executed_notebook.ipynb` preserves the original executed notebook used to initialize this repository. The clean notebook under `notebooks/` fixes the original hidden-kernel-state dependency and can run from a fresh Python process.

