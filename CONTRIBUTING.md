# Contributing

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Suggested improvement workflow

1. Create a focused branch.
2. Add or update tests before changing the generator.
3. Keep synthetic assumptions explicit in `docs/methodology.md`.
4. Run `pytest` and `python scripts/reproduce.py`.
5. Do not present synthetic performance as evidence of clinical validity.

Useful future contributions include calibration, nested cross-validation, uncertainty estimates, class-conditional error analysis, model cards, explainability, and a clearer external-validation plan.

