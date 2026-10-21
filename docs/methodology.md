# Methodology and assumptions

## Objective

This project creates a synthetic tabular cohort for demonstrating multi-class classification of respiratory and hypoxia-related patterns. It prioritizes reproducibility, interpretability, and physiologically plausible relationships over clinical realism.

## Cohort design

The default cohort contains 2,150 observations across seven intentionally imbalanced classes. A latent severity value sampled from a beta distribution introduces within-class heterogeneity. Additional random noise and overlapping distributions reduce trivial separability.

## Physiological structure

The generator uses simplified relationships for:

- alveolar oxygen from inspired oxygen and arterial carbon dioxide;
- an age-dependent alveolar–arterial gradient;
- arterial saturation from a smooth oxygen-dissociation approximation;
- oxygen content from hemoglobin, saturation, and dissolved oxygen;
- acid–base coupling through bicarbonate and carbon dioxide;
- obstruction through airway resistance and FEV1/FVC;
- diffusion through DLCO;
- pulmonary congestion through B-lines, chest X-ray score, and BNP.

Several intermediate variables—such as barometric pressure, alveolar oxygen, arterial oxygen, and the A–a gradient—are not included as model features. This reduces direct shortcuts between the generating mechanism and the target label.

## Modeling

The notebook uses a stratified train/test split and compares:

- Decision Tree;
- Gaussian Naive Bayes;
- Random Forest.

Grid search is repeated for accuracy, macro precision, macro recall, and macro F1. Macro-averaged metrics are emphasized because the class distribution is imbalanced. Confusion matrices and permutation importance support interpretation.

## Important limitations

This is a closed synthetic system. The labels and features share a generating process, so measured performance is expected to be optimistic. The project does not establish diagnostic validity, causal effects, fairness, calibration in clinical populations, or safety.

