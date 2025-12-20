# Preference density × mechanism: an LLM-enabled regime shift test
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/llm-hayek-roth)

So what: this repo runs a minimal, reproducible 2×2 “field-experiment analog” showing how **LLM-based elicitation** can make preferences *easier to infer* and lower the ROI threshold (λ*) at which **centralized recommendations/matching** becomes worthwhile relative to decentralized search.

If you only read one thing: start at `reports/paper_latest/README.md` (one table + key figures).

## At a glance

**Motivated lay reader**

Search is the “backup plan” when the platform can’t confidently pick the right match for you.
We make you scroll/filter/message because the system doesn’t really understand what you want.

This repo tests a mechanism-design claim: if an AI intake can reliably turn messy human intent into a structured spec, then the platform can move from “you search” to “we recommend/match” in categories where search is costly.

The cool part is that this can be a threshold story, not a smooth one: better preference inference can flip which mechanism is welfare‑dominant (and when it’s worth investing in centralized matching).

**Economist**

Inspired by Peng Shi, “Optimal Matchmaking Strategy in Two-sided Marketplaces” (SSRN 3536086; see References): mechanism choice depends on preference density/inferability and attention/communication costs.

Operationally, we do:
- A 2×2: {standard form vs LLM elicitation} × {decentralized search vs centralized recommendations/matching}.
- A regime sweep over elicitation depth (`k_I × k_J`) that plots net-welfare differences and the implied break-even boundary `λ*` (central wins if `λ > λ*`).
- Stress tests: congestion via delegated outreach, field-style cluster randomization + clustered inference, and a richer FieldSim v2 with pricing/dynamics/cancellations/spillovers.

Quick definitions:
- `attention_cost` (λ): per-action cost charged for communication/attention.
- `λ*`: break-even λ where central and search tie in net welfare; central wins if `λ > λ*`.
- `k_I` / `k_J`: elicitation depth for customers / providers (how much structured preference signal you extract).
- `d_hat`: preference-density proxy (how predictable acceptances are, given inferred preferences).

What we measure:
- How accurately the platform can infer who you’ll accept (a proxy for “describability” / preference density).
- How much communication/attention is spent to get a match.
- Net welfare under each mechanism, and the implied ROI boundary `λ*` where centralized matching starts to beat search.

**Framing**
- Not “LLMs always improve markets” (effects vary by regime).
- Yes “LLMs can move you across a threshold where central mechanisms become viable,” summarized by the regime map ROI boundary `λ*` and the hard-category interaction.

## Results (inline)

Headline results (from `reports/paper_latest/`):
- FieldSim v2 (hard): net welfare per customer is highest in `ai_central` (18.58) vs 14.13–14.48 in the other arms.
- Regime sweep (hard): `λ*` falls from 0.014 (`k_I=1,k_J=1`) to 0.0123 (`k_I=6,k_J=6`).
- Delegated outreach congestion: saturation 1.0 vs 0.0 reduces net welfare per customer by ~0.883 (hard).

Net welfare per customer (FieldSim v2, hard; higher is better):

| intake | search | central |
| --- | --- | --- |
| standard | 14.13 | 14.48 |
| LLM | 14.27 | 18.58 |

Full table: `reports/paper_latest/key_results.md`.

![Regime map: ROI boundary λ* (hard)](reports/paper_latest/fig_regime_map_hard_lambda_star.svg)
![Regime map: Central − Search net welfare (hard)](reports/paper_latest/fig_regime_map_hard_net_welfare_diff.svg)
![FieldSim v2: net welfare by arm (hard)](reports/paper_latest/fig_hard_net_welfare.svg)

<details>
<summary>More plots</summary>

![Delegated outreach: net welfare vs saturation (hard)](reports/paper_latest/fig_hard_net_welfare_vs_saturation.svg)
![FieldSim v2: reciprocity proxy vs rank (hard)](reports/paper_latest/fig_hard_reciprocity_curve.svg)

</details>

**Money/prices**

The “realistic” simulation layer (`make field-v2`) includes explicit budgets, quotes, cancellations, and surplus/profit accounting (GPT is used for parsing/elicitation only).

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
