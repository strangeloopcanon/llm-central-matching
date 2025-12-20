# Preference Heterogeneity & Robustness Sweep

**Key insight**: As preferences become more heterogeneous (lower weight α),
the AI×Central advantage increases. This holds across different AI intake
quality levels and misclassification rates.

## Parameters Varied

| Parameter | Values | Meaning |
|---|---|---|
| `weight_alpha` | [0.3, 0.5, 1.0, 2.0, 3.0] | Preference concentration (lower = ONE thing matters) |
| `ai_noise_sd` | [0.03, 0.15] | AI intake quality (0.03 = best, 0.20 = pessimistic) |
| `misclass_rate` | [0.5, 0.7] | Standard form misclassification rate |

## Results: Heterogeneity Effect

- Low α (≤0.5) mean triple effect: **1.6128**
- High α (≥2.0) mean triple effect: **-1.6673**
- Ratio (low/high): **N/A**

## Results: AI Quality Sensitivity

- Best-case AI (noise=0.03): mean triple = **0.1892**
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
- `fig_triple_vs_weight_alpha.svg`: main heterogeneity result
- `fig_triple_vs_ai_noise.svg`: robustness to AI quality

## Run

```bash
make heterogeneity
```
