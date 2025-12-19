# Preference density × mechanism: an LLM-enabled regime shift test
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/llm-hayek-roth)

So what: this repo runs a minimal, reproducible 2×2 “field-experiment analog” showing how **LLM-based elicitation** can make preferences *easier to infer* and lower the ROI threshold (λ*) at which **centralized recommendations/matching** becomes worthwhile relative to decentralized search.

<details>
<summary>Why this matters</summary>

This project is inspired by Peng Shi, “Optimal Matchmaking Strategy in Two-sided Marketplaces” (SSRN 3536086; see References).

Search is the “backup plan” when the platform can’t confidently pick the right match for you.
We make you scroll/filter/message because the system doesn’t really understand what you want.

This repo tests a mechanism-design claim: if an AI intake can reliably turn messy human intent into a structured spec, then the platform can move from “you search” to “we recommend/match” in categories where search is costly.

What we measure:
- How accurately the platform can infer who you’ll accept (a proxy for “describability”).
- How much communication/attention is spent to get a match.
- The ROI boundary `λ*` where centralized matching starts to beat search (central wins if `λ > λ*`).

Money/prices:
- The “realistic” simulation layer (`make field-v2`) includes explicit budgets, quotes, and surplus/profit accounting.

If you only read one output: start at `reports/paper_latest/README.md`.

</details>

## One-command run

1) Ensure `.env` contains `OPENAI_API_KEY`.
2) Run:

```bash
make setup
make experiment
```

Outputs land in `reports/latest/`:
- `reports/latest/summary_table.csv`
- `reports/latest/summary_table.md`
- `reports/latest/effects_table.csv`
- `reports/latest/effects_table.md`
- `reports/latest/fig_*.svg`
- `reports/latest/run_metadata.json`
- `reports/latest/README.md`

<details>
<summary>Interface Contract commands</summary>

```bash
make check
make test
make llm-live
make all
```

</details>

<details>
<summary>Regime map (k_I × k_J)</summary>

```bash
make regime
```

This sweeps elicitation depth for customers (k_I) and providers (k_J) and plots where
centralized recommendations beat search on net welfare, and the implied ROI boundary (λ*).

- `reports/latest/regime_map_hard_net_welfare_diff.svg`: Central − Search net welfare at the configured `attention_cost`
- `reports/latest/regime_map_hard_lambda_star.svg`: threshold λ* where central starts to win (central wins if `λ > λ*`)
- `reports/latest/regime_grid_hard.csv`: underlying grid (includes `lambda_star`)

</details>

<details>
<summary>Experiment 2: Delegated outreach × congestion (saturation)</summary>

```bash
make congestion
```

Outputs land in `reports/latest_congestion/`:
- `reports/latest_congestion/congestion_saturation_easy.md`
- `reports/latest_congestion/congestion_saturation_hard.md`
- `reports/latest_congestion/fig_*_vs_saturation.svg`
- `reports/latest_congestion/congestion_meta_*.json`

</details>

<details>
<summary>Field-style simulation: cluster randomization + clustered inference</summary>

```bash
make field
```

Outputs land in `reports/field_latest/`:
- `reports/field_latest/arm_summary.md`
- `reports/field_latest/reg_matched.md`
- `reports/field_latest/reg_total_value.md`
- `reports/field_latest/fig_*`
- `reports/field_latest/README.md`

</details>

<details>
<summary>FieldSim v2: dynamics + pricing + cancellations + spillovers + scaling</summary>

```bash
make field-v2
```

Outputs land in `reports/field_v2_latest/`:
- `reports/field_v2_latest/arm_summary.md`
- `reports/field_v2_latest/reg_matched.md`
- `reports/field_v2_latest/spillovers.md`
- `reports/field_v2_latest/fig_*`
- `reports/field_v2_latest/scaling_summary.csv`
- `reports/field_v2_latest/README.md`

</details>

<details>
<summary>Intake verisimilitude samples (home services)</summary>

```bash
make intakes
```

Outputs land in `reports/intakes_latest/`:
- `reports/intakes_latest/intake_samples.json`
- `reports/intakes_latest/README.md`

</details>

<details>
<summary>FieldSim v2 calibration (moment matching)</summary>

```bash
make calibrate
```

Outputs land in `reports/field_v2_calibration_latest/`:
- `reports/field_v2_calibration_latest/best_params.json`
- `reports/field_v2_calibration_latest/best_moments.json`
- `reports/field_v2_calibration_latest/fig_loss.svg`
- `reports/field_v2_calibration_latest/README.md`

To run FieldSim v2 with calibrated params:

```bash
make field-v2-calibrated
```

Outputs land in `reports/field_v2_calibrated_latest/`.

</details>

<details>
<summary>LLM ablations: bits vs parsing vs agent</summary>

```bash
make ablations
```

Outputs land in `reports/ablations_latest/`:
- `reports/ablations_latest/parsing_quality_easy.md`
- `reports/ablations_latest/parsing_quality_hard.md`
- `reports/ablations_latest/ablations_summary_easy.md`
- `reports/ablations_latest/ablations_summary_hard.md`
- `reports/ablations_latest/fig_*`
- `reports/ablations_latest/README.md`

</details>

<details>
<summary>FieldSim v2 sensitivity sweep (seeds × knobs)</summary>

```bash
make field-v2-sensitivity
```

Outputs land in `reports/field_v2_sensitivity_latest/`:
- `reports/field_v2_sensitivity_latest/runs.csv`
- `reports/field_v2_sensitivity_latest/summary.json`
- `reports/field_v2_sensitivity_latest/fig_triple_welfare_vs_rec_k.svg`
- `reports/field_v2_sensitivity_latest/README.md`

</details>

<details>
<summary>Paper bundle (single entrypoint table + key figures)</summary>

```bash
make paper-bundle
```

Outputs land in `reports/paper_latest/`:
- `reports/paper_latest/README.md`
- `reports/paper_latest/key_results.md`
- `reports/paper_latest/fig_*.svg`

</details>

## References

- Peng Shi, “Optimal Matchmaking Strategy in Two-sided Marketplaces” (SSRN 3536086). Optional local PDF (gitignored): `Optimal Matchmaking Strategy in Two-sided Marketplaces.pdf` (see also https://ssrn.com/abstract=3536086).
- Selected related marketplace design evidence referenced in Shi: congestion in matching markets (Roth and co-authors), preference signaling (e.g., Coles et al., Lee & Niederle), and platform experiments on recommendations/search frictions (e.g., Fradkin; Horton; Li & Netessine).
