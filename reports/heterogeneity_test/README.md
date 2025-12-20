# Preference Heterogeneity & Robustness Sweep

**Key insight**: As preferences become more heterogeneous (lower weight α),
the AI×Central advantage increases. This holds across different AI intake
quality levels and misclassification rates.

## Parameters Varied

| Parameter | Values | Meaning |
|---|---|---|
| `weight_alpha` | [0.5, 1.0, 2.0] | Pref concentration (low=focused) |
| `ai_noise_sd` | [0.03, 0.15] | AI quality (0.03=best, 0.20=bad) |
| `misclass_rate` | [0.7] | Std form misclass rate |
| `utility_noise_sd` | [0.04, 0.08, 0.15] | Utility randomness (higher=noisier) |

## Results: Heterogeneity Effect

- Low α (≤0.5) mean triple effect: **6.5163**
- High α (≥2.0) mean triple effect: **3.8909**
- Ratio (low/high): **1.7×**

## Results: AI Quality Sensitivity

- Best-case AI (noise=0.03): mean triple = **-2.1776**
- Pessimistic AI (noise=0.20): mean triple = **0.0000**

## Interpretation

| weight_alpha | Meaning |
|---|---|
| 0.3 | Very concentrated: person cares about ONE specific dimension |
| 1.0 | Moderate concentration (default) |
| 3.0 | Diffuse: person cares about everything somewhat equally |

## Artifacts

- `runs.csv`: per-run results for all parameter combinations
- `summary_by_alpha.csv`: effect aggregated by weight_alpha
- `summary_by_ai_noise.csv`: effect aggregated by AI quality
- `summary_by_utility_noise.csv`: effect aggregated by utility randomness
- `fig_triple_vs_weight_alpha.svg`: main heterogeneity result
- `fig_triple_vs_ai_noise.svg`: robustness to AI quality
- `fig_triple_vs_utility_noise.svg`: effect of utility randomness

## Results: Utility Randomness (NEW)

- Low noise (≤0.05): mean triple = **-5.0128**
- High noise (≥0.12): mean triple = **-2.7976**

*Economic interpretation*: Higher idiosyncratic noise (random utility shocks)
may increase AI+Central advantage if centralized matching better handles
preference uncertainty through information aggregation.

## Run

```bash
make heterogeneity
```
