# Latest run

Mechanism: central = deferred acceptance (stable matching), proposer=customer.

So what: Roth-style benchmark (stable matching under hat-based rankings).
We apply a mutual acceptability cutoff, and centralization is very attention-light.
Focus on the net-welfare gap and the ROI boundary (λ*).

Key diagnostics (match_rate DiD):
- easy: d̂_I 0.804 → 0.803; d̂_J 0.801 → 0.803
  central match_rate 0.619 → 0.619; DiD +0.000
  net_welfare/customer (central - search): standard +0.101; ai +0.101; DiD +0.000
- hard: d̂_I 0.795 → 0.802; d̂_J 0.794 → 0.799
  central match_rate 0.603 → 0.603; DiD -0.001
  net_welfare/customer (central - search): standard +0.108; ai +0.107; DiD -0.001

ROI boundary (λ*): central beats search if λ > λ*.
- easy: λ* standard=0.0004, ai=0.0004
- hard: λ* standard=0.0004, ai=0.0004

Artifacts:
- `summary_table.csv` / `summary_table.md`: arm-by-arm outcomes
- `effects_table.csv` / `effects_table.md`: key contrasts (DiD, etc.)
- `fig_*`: quick plots by category
- `run_metadata.json`: parameters + seeds

Optional regime map:
- Run `make regime` to write:
  - `regime_map_hard_net_welfare_diff.svg` (at the configured attention_cost)
  - `regime_map_hard_lambda_star.svg` (ROI boundary λ*; central wins if λ > λ*)
  - `regime_grid_hard.csv` (underlying grid)

Interpretation tip: look for the interaction—`ai_central` beats `standard_central`
by more than `ai_search` beats `standard_search`.

Figures (easy):
- `fig_easy_match_rate.svg`
- `fig_easy_total_value.svg`
- `fig_easy_d_hat_I.svg`
- `fig_easy_d_hat_J.svg`
- `fig_easy_attention_per_match.svg`
- `fig_easy_net_welfare_per_customer.svg`

Figures (hard):
- `fig_hard_match_rate.svg`
- `fig_hard_total_value.svg`
- `fig_hard_d_hat_I.svg`
- `fig_hard_d_hat_J.svg`
- `fig_hard_attention_per_match.svg`
- `fig_hard_net_welfare_per_customer.svg`

