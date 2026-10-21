# Data dictionary

| Variable | Unit / type | Description |
|---|---:|---|
| `Age` | years | Simulated patient age. |
| `HR` | beats/min | Heart rate. |
| `RR` | breaths/min | Respiratory rate. |
| `SpO2` | % | Pulse-oximetry oxygen saturation. |
| `PaCO2` | mmHg | Arterial carbon-dioxide partial pressure. |
| `HCO3` | mEq/L | Bicarbonate concentration. |
| `pH` | unitless | Arterial pH. |
| `Hb` | g/dL | Hemoglobin concentration. |
| `CaO2` | mL O₂/dL | Arterial oxygen content. |
| `AirwayResistance` | synthetic scale | Proxy for airway obstruction. |
| `FEV1_FVC` | ratio | Spirometric obstruction/restriction indicator. |
| `DLCO_percent` | % predicted | Simulated diffusion capacity. |
| `B_lines` | count | Lung-ultrasound B-lines. |
| `CXR_score` | 0–12 | Synthetic chest X-ray severity score. |
| `BNP` | pg/mL | Congestion-related biomarker. |
| `label` | category | One of seven synthetic outcome classes. |

## Outcome classes

| Label | Intended synthetic pattern |
|---|---|
| `healthy` | Reference physiology with minimal abnormalities. |
| `hypoxia_altitude` | Reduced inspired oxygen with a small A–a gradient. |
| `hypoxia_anemic` | Low hemoglobin with relatively preserved pulmonary oxygenation. |
| `asthma` | Obstructive pattern with variable V/Q mismatch. |
| `edema` | Congestion, B-lines, imaging changes, and impaired oxygenation. |
| `fibrosis` | Restrictive/diffusion-limited interstitial pattern. |
| `emphysema` | Obstruction, reduced diffusion, and possible CO₂ retention. |

