"""Synthetic, physiology-inspired pulmonary dataset generation.

The generator is deterministic for a fixed random seed. The data are intended
for education and model-development demonstrations, not clinical use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _spo2_from_pao2(pao2: np.ndarray) -> np.ndarray:
    """
    Approx Severinghaus-like saturation curve.
    Key property: PaO2 ~ 60 mmHg -> SpO2 ~ ~90% (not 50%).
    """
    pao2 = np.clip(pao2, 1e-3, 200)
    return 100.0 * (pao2**3 + 150.0 * pao2) / (pao2**3 + 150.0 * pao2 + 23400.0)


def generate_pulmonary_synthetic_dataset(n_per_class=None, seed: int = 42) -> pd.DataFrame:
    """
    Create a multi-class pulmonary dataset with realistic ranges, randomness, and overlap.
    Labels:
      - healthy
      - hypoxia_altitude
      - hypoxia_anemic
      - asthma
      - edema
      - fibrosis
      - emphysema

    Output columns (15 features + label):
      Age, HR, RR, SpO2, PaCO2, HCO3, pH, Hb, CaO2,
      AirwayResistance, FEV1_FVC, DLCO_percent, B_lines, CXR_score, BNP, label
    """
    if n_per_class is None:
        # Intentionally imbalanced (more healthy), to avoid trivial accuracy-only evaluation later
        n_per_class = {
            "healthy": 900,
            "hypoxia_altitude": 250,
            "hypoxia_anemic": 220,
            "asthma": 260,
            "edema": 180,
            "fibrosis": 160,
            "emphysema": 180,
        }

    rng = np.random.default_rng(seed)
    rows = []

    for label, n in n_per_class.items():
        # 1) Latent severity drives within-class variability (skewed toward mild)
        severity = rng.beta(2, 4.5, size=n)
        if label in ("edema", "emphysema", "fibrosis"):
            severity = np.clip(severity * 1.15 + 0.05, 0, 1)
        elif label == "asthma":
            severity = np.clip(severity * 1.05 + 0.02, 0, 1)

        # 2) Age: label-dependent but overlapping (no hard separation)
        age_mu = {
            "healthy": 35, "hypoxia_altitude": 28, "hypoxia_anemic": 46,
            "asthma": 30, "edema": 72, "fibrosis": 66, "emphysema": 69
        }[label]
        age_sd = {
            "healthy": 14, "hypoxia_altitude": 10, "hypoxia_anemic": 16,
            "asthma": 13, "edema": 11, "fibrosis": 11, "emphysema": 10
        }[label]
        Age = np.clip(rng.normal(age_mu, age_sd, size=n), 18, 90)

        # 3) Internal barometric pressure Pbar (mmHg) for alveolar gas equation
        # NOTE: NOT stored as a feature (prevents "altitude is obvious" shortcuts).
        if label == "hypoxia_altitude":
            Pbar = np.clip(rng.normal(610, 35, size=n), 520, 700)  # moderate altitude range
        else:
            Pbar = np.clip(rng.normal(760, 12, size=n), 700, 790)

        # 4) Latent physiology factors (overlap + noise)
        # obstruction -> affects airway metrics + CO2 retention + spirometry
        # lung_water  -> affects B-lines/CXR + oxygenation
        # diffusion   -> affects DLCO + oxygenation (fibrosis/emphysema)
        obstruction_base = {
            "healthy": 0.10, "hypoxia_altitude": 0.12, "hypoxia_anemic": 0.12,
            "asthma": 0.60, "edema": 0.14, "fibrosis": 0.18, "emphysema": 0.65
        }[label]
        obstruction = np.clip(obstruction_base + 0.60 * severity + rng.normal(0, 0.12, size=n), 0, 1.2)

        lung_water_base = {
            "healthy": 0.08, "hypoxia_altitude": 0.10, "hypoxia_anemic": 0.10,
            "asthma": 0.12, "edema": 0.78, "fibrosis": 0.42, "emphysema": 0.12
        }[label]
        lung_water = np.clip(lung_water_base + 0.55 * severity + rng.normal(0, 0.12, size=n), 0, 1.3)

        diffusion_base = {
            "healthy": 0.10, "hypoxia_altitude": 0.12, "hypoxia_anemic": 0.12,
            "asthma": 0.12, "edema": 0.45, "fibrosis": 0.80, "emphysema": 0.60
        }[label]
        diffusion = np.clip(diffusion_base + 0.55 * severity + rng.normal(0, 0.10, size=n), 0, 1.4)

        # 5) PaCO2 (mmHg): ventilation patterns (hyperventilation vs retention)
        if label == "healthy":
            PaCO2 = rng.normal(40, 3.0, size=n)
        elif label == "hypoxia_altitude":
            PaCO2 = rng.normal(34, 3.2, size=n) - 2.5 * severity  # hyperventilation
        elif label == "hypoxia_anemic":
            PaCO2 = rng.normal(39, 3.2, size=n) - 0.8 * severity
        elif label == "fibrosis":
            PaCO2 = rng.normal(36, 3.2, size=n) - 3.5 * severity  # chronic hyperventilation tendency
        elif label == "edema":
            PaCO2 = rng.normal(38, 3.8, size=n) + 2.0 * (severity ** 1.5)  # some retention in severe
        elif label == "asthma":
            # mild: can be low; severe: retention can appear
            PaCO2 = 34 + 1.5 * severity + 12.0 * (severity ** 2) + rng.normal(0, 4.0, size=n)
        elif label == "emphysema":
            PaCO2 = 42 + 9.0 * (severity ** 1.2) + rng.normal(0, 4.2, size=n)
        PaCO2 = np.clip(PaCO2, 24, 75)

        # 6) RR and HR: severity-driven with overlap
        RR_base = {
            "healthy": 14, "hypoxia_altitude": 17, "hypoxia_anemic": 16,
            "asthma": 16, "edema": 20, "fibrosis": 19, "emphysema": 18
        }[label]
        RR = np.clip(RR_base + 9.0 * severity + rng.normal(0, 2.0, size=n), 10, 40)

        HR_base = {
            "healthy": 70, "hypoxia_altitude": 78, "hypoxia_anemic": 82,
            "asthma": 78, "edema": 88, "fibrosis": 84, "emphysema": 86
        }[label]
        HR = np.clip(HR_base + 26.0 * severity + rng.normal(0, 8.5, size=n), 45, 160)

        # 7) Hemoglobin Hb (g/dL): anemia strongest in hypoxia_anemic, overlap elsewhere
        if label == "hypoxia_anemic":
            Hb = rng.normal(9.6, 1.3, size=n) + 0.4 * rng.normal(0, 1, size=n)
        elif label == "hypoxia_altitude":
            Hb = rng.normal(14.2, 1.2, size=n) + 0.7 * severity  # mild polycythemia tendency
        elif label in ("emphysema", "fibrosis", "edema"):
            Hb = rng.normal(12.9, 1.4, size=n) + 0.3 * rng.normal(0, 1, size=n)
        else:
            Hb = rng.normal(14.0, 1.3, size=n)
        Hb = np.clip(Hb, 6.5, 18.5)

        # 8) A–a gradient (mmHg): age component + pathology component (internal; not stored)
        AA_expected = (Age / 4.0) + 4.0
        add_AA = {
            "healthy": 0.0,
            "hypoxia_altitude": 1.5,   # lungs are OK: small A-a
            "hypoxia_anemic": 2.0,     # oxygenation OK, mild noise
            "asthma": 9.0,             # V/Q mismatch
            "edema": 22.0,             # shunt/VQ
            "fibrosis": 18.0,          # diffusion limitation
            "emphysema": 12.0          # V/Q + surface loss
        }[label]
        A_a = AA_expected + add_AA * (0.35 + 0.85 * severity) + rng.normal(0, 3.5, size=n)
        A_a = A_a + rng.normal(0, 2.0, size=n) * (rng.random(n) < 0.06)  # occasional noise/outliers
        A_a = np.clip(A_a, 0, 65)

        # 9) Alveolar gas equation (internal): PAO2 -> PaO2 -> SpO2
        FiO2, PH2O, RQ = 0.21, 47.0, 0.8
        PiO2 = FiO2 * (Pbar - PH2O)
        PAO2 = PiO2 - (PaCO2 / RQ)

        PaO2 = np.clip(PAO2 - A_a + rng.normal(0, 5.0, size=n), 30, 120)

        # Realistic SpO2 curve (key fix)
        SpO2 = _spo2_from_pao2(PaO2) + rng.normal(0, 1.6, size=n)
        SpO2 = np.clip(SpO2, 55, 100)

        # 10) HCO3 (mEq/L) + pH from Henderson–Hasselbalch
        if label == "emphysema":
            HCO3 = rng.normal(27.0, 2.2, size=n) + 3.0 * (severity ** 1.1)  # chronic compensation
        elif label == "asthma":
            HCO3 = rng.normal(24.0, 2.1, size=n) + 1.0 * (severity ** 1.2)
        elif label in ("hypoxia_altitude", "fibrosis"):
            HCO3 = rng.normal(22.8, 1.9, size=n) - 0.9 * severity
        else:
            HCO3 = rng.normal(24.2, 2.0, size=n) + 0.6 * (severity ** 1.1)
        HCO3 = np.clip(HCO3, 14, 40)

        pH = 6.1 + np.log10(HCO3 / (0.03 * PaCO2)) + rng.normal(0, 0.02, size=n)
        pH = np.clip(pH, 7.02, 7.58)

        # 11) Oxygen content CaO2 (mL O2/dL): combines Hb + saturation
        CaO2 = 1.34 * Hb * (SpO2 / 100.0) + 0.003 * PaO2 + rng.normal(0, 0.35, size=n)
        CaO2 = np.clip(CaO2, 4.5, 24.5)

        # 12) Airway resistance and FEV1/FVC (class-specific realism fix)
        AirwayResistance = np.clip(1.6 + 4.6 * obstruction + rng.normal(0, 0.7, size=n), 0.8, 10.0)

        if label in ("asthma", "emphysema"):
            FEV1_FVC = 0.78 - 0.35 * obstruction + rng.normal(0, 0.05, size=n)
        elif label == "fibrosis":
            FEV1_FVC = 0.84 - 0.08 * obstruction + rng.normal(0, 0.04, size=n)
        else:
            # healthy / altitude / anemic / edema: should stay mostly normal
            FEV1_FVC = 0.80 - 0.12 * obstruction + rng.normal(0, 0.04, size=n)

        FEV1_FVC = np.clip(FEV1_FVC, 0.30, 0.92)

        # 13) DLCO % predicted: reduced in fibrosis/emphysema; asthma typically normal-ish
        DLCO_base = {
            "healthy": 92, "hypoxia_altitude": 90, "hypoxia_anemic": 88,
            "asthma": 95, "edema": 70, "fibrosis": 50, "emphysema": 45
        }[label]
        DLCO_percent = np.clip(DLCO_base - 35.0 * diffusion + rng.normal(0, 10.0, size=n), 18, 125)

        # 14) Imaging surrogates: B-lines and CXR score driven by lung_water + overlap/noise
        lam = 0.8 + 10.0 * lung_water
        lam = lam + 3.0 * (rng.random(n) < 0.06)  # occasional artifacts
        B_lines = np.clip(rng.poisson(lam=np.clip(lam, 0.2, 22.0)), 0, 30)

        CXR_mean = 0.8 + 7.5 * lung_water + 2.0 * diffusion + rng.normal(0, 1.2, size=n)
        CXR_score = np.clip(np.rint(CXR_mean), 0, 12).astype(int)

        # 15) BNP (pg/mL): edema strongest but overlaps with age/comorbidity
        if label == "edema":
            bnp_log_mu, bnp_log_sd = 5.5 + 0.65 * severity, 0.55
        elif label in ("fibrosis", "emphysema"):
            bnp_log_mu, bnp_log_sd = 4.6 + 0.30 * severity + 0.004 * (Age - 50), 0.60
        else:
            bnp_log_mu, bnp_log_sd = 3.9 + 0.18 * severity + 0.002 * (Age - 40), 0.65

        BNP = np.clip(rng.lognormal(mean=bnp_log_mu, sigma=bnp_log_sd, size=n), 5, 3500)

        rows.append(pd.DataFrame({
            "Age": Age,
            "HR": HR,
            "RR": RR,
            "SpO2": SpO2,
            "PaCO2": PaCO2,
            "HCO3": HCO3,
            "pH": pH,
            "Hb": Hb,
            "CaO2": CaO2,
            "AirwayResistance": AirwayResistance,
            "FEV1_FVC": FEV1_FVC,
            "DLCO_percent": DLCO_percent,
            "B_lines": B_lines,
            "CXR_score": CXR_score,
            "BNP": BNP,
            "label": label
        }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
