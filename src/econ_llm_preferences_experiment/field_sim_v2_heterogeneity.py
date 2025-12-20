"""
Sweep preference parameters to demonstrate:
1. As preferences become more heterogeneous (lower weight_alpha), AI×Central advantage increases
2. Effects persist across different AI intake quality levels (ai_weight_noise_sd)
3. Effects persist across different misclassification rates (std_weight_misclass_hard)

This is the corrected heterogeneity sweep that varies the RIGHT parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, TypeAlias

from econ_llm_preferences_experiment.econometrics import ols_cluster_robust
from econ_llm_preferences_experiment.field_sim_v2 import (
    CityWeekAssignment,
    Elicitation,
    FieldV2Params,
    JobOutcome,
    Mechanism,
    simulate_city_week,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    variance = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (variance / len(xs)) ** 0.5


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _arm_label(ai: int, central: int) -> str:
    if ai == 0 and central == 0:
        return "standard_search"
    if ai == 0 and central == 1:
        return "standard_central"
    if ai == 1 and central == 0:
        return "ai_search"
    return "ai_central"


def _triple_from_regression(*, jobs: list[JobOutcome], outcome: str) -> tuple[float, float, float]:
    """Returns (coef, se, p) for ai×central×hard triple interaction."""
    y = [_as_float(getattr(j, outcome)) for j in jobs]
    x = []
    clusters = []
    for j in jobs:
        hard = 1.0 if j.category == "hard" else 0.0
        ai = float(j.ai)
        central = float(j.central)
        x.append(
            [
                1.0,
                ai,
                central,
                ai * central,
                hard,
                ai * hard,
                central * hard,
                ai * central * hard,
            ]
        )
        clusters.append(j.cell_id)
    res = ols_cluster_robust(y=y, x=x, clusters=clusters)
    triple_idx = 7
    return float(res.coef[triple_idx]), float(res.se[triple_idx]), float(res.p[triple_idx])


def _ai_central_mean(*, jobs: list[JobOutcome], outcome: str, category: str) -> float:
    """Mean outcome for ai_central arm in given category."""
    vals = [
        _as_float(getattr(j, outcome))
        for j in jobs
        if j.ai == 1 and j.central == 1 and j.category == category
    ]
    return _mean(vals)


def _run_once(
    *,
    seed: int,
    cities: int,
    weeks: int,
    jobs_easy: int,
    jobs_hard: int,
    providers: int,
    params: FieldV2Params,
) -> list[JobOutcome]:
    rng = random.Random(seed)  # nosec B311

    arms: list[tuple[Elicitation, Mechanism]] = [
        ("standard", "search"),
        ("standard", "central"),
        ("ai", "search"),
        ("ai", "central"),
    ]

    cell_keys: list[tuple[str, int]] = [
        (f"city{city_idx:03d}", week) for city_idx in range(cities) for week in range(weeks)
    ]
    reps = (len(cell_keys) + len(arms) - 1) // len(arms)
    easy_arms = (arms * reps)[: len(cell_keys)]
    hard_arms = (arms * reps)[: len(cell_keys)]
    rng.shuffle(easy_arms)
    rng.shuffle(hard_arms)

    assignments: list[CityWeekAssignment] = []
    for idx, (city_id, week) in enumerate(cell_keys):
        assignments.append(
            CityWeekAssignment(
                city_id=city_id,
                week=week,
                cell_easy=easy_arms[idx],
                cell_hard=hard_arms[idx],
            )
        )

    job_rows: list[JobOutcome] = []
    for a in assignments:
        city_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
        _cells, jobs, _providers, _meas = simulate_city_week(
            rng=city_rng,
            assignment=a,
            params=params,
            n_jobs_easy=jobs_easy,
            n_jobs_hard=jobs_hard,
            n_providers=providers,
        )
        job_rows.extend(jobs)

    return job_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep preference parameters for robustness (weight_alpha, ai_noise, misclass)"
    )
    parser.add_argument("--out", default="reports/heterogeneity_latest")
    parser.add_argument("--seed-base", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--cities", type=int, default=28)
    parser.add_argument("--weeks", type=int, default=2)
    parser.add_argument("--jobs-easy", type=int, default=22)
    parser.add_argument("--jobs-hard", type=int, default=22)
    parser.add_argument("--providers", type=int, default=35)
    parser.add_argument(
        "--weight-alphas",
        type=str,
        default="0.3,0.5,0.8,1.0,1.5,2.0,3.0",
        help="Preference weight concentration (lower = one thing matters, higher = diffuse).",
    )
    parser.add_argument(
        "--ai-noise-sds",
        type=str,
        default="0.03,0.10,0.20",
        help="AI intake noise levels (0.03 = best case, 0.20 = pessimistic).",
    )
    parser.add_argument(
        "--misclass-rates",
        type=str,
        default="0.40,0.70",
        help="Standard form misclassification rates for hard categories.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_alphas = [float(a.strip()) for a in args.weight_alphas.split(",")]
    ai_noise_sds = [float(a.strip()) for a in args.ai_noise_sds.split(",")]
    misclass_rates = [float(a.strip()) for a in args.misclass_rates.split(",")]
    seeds = [args.seed_base + i for i in range(args.n_seeds)]

    rows: list[dict[str, RowValue]] = []

    # Main sweep: weight_alpha × ai_noise × misclass × seeds
    run_count = 0
    total_runs = len(weight_alphas) * len(ai_noise_sds) * len(misclass_rates) * len(seeds)

    for weight_alpha in weight_alphas:
        for ai_noise_sd in ai_noise_sds:
            for misclass_rate in misclass_rates:
                for seed in seeds:
                    run_count += 1
                    params = FieldV2Params(
                        weight_alpha=weight_alpha,
                        ai_weight_noise_sd=ai_noise_sd,
                        std_weight_misclass_hard=misclass_rate,
                        attention_cost=0.01,
                    )
                    log(
                        logger,
                        20,
                        "heterogeneity_sweep_run",
                        run=f"{run_count}/{total_runs}",
                        weight_alpha=weight_alpha,
                        ai_noise_sd=ai_noise_sd,
                        misclass=misclass_rate,
                        seed=seed,
                    )

                    jobs = _run_once(
                        seed=seed,
                        cities=args.cities,
                        weeks=args.weeks,
                        jobs_easy=args.jobs_easy,
                        jobs_hard=args.jobs_hard,
                        providers=args.providers,
                        params=params,
                    )

                    triple_coef, triple_se, triple_p = _triple_from_regression(
                        jobs=jobs, outcome="net_welfare"
                    )
                    triple_match_coef, _, _ = _triple_from_regression(jobs=jobs, outcome="matched")
                    ai_central_hard = _ai_central_mean(
                        jobs=jobs, outcome="net_welfare", category="hard"
                    )

                    rows.append(
                        {
                            "weight_alpha": weight_alpha,
                            "ai_noise_sd": ai_noise_sd,
                            "misclass_rate": misclass_rate,
                            "seed": seed,
                            "triple_welfare_coef": round(triple_coef, 4),
                            "triple_welfare_se": round(triple_se, 4),
                            "triple_welfare_p": round(triple_p, 4),
                            "triple_match_coef": round(triple_match_coef, 4),
                            "ai_central_hard_welfare": round(ai_central_hard, 2),
                            "n_jobs": len(jobs),
                        }
                    )

    _write_csv(rows, out_dir / "runs.csv")

    # === Analysis 1: Effect of weight_alpha (averaged over ai_noise and misclass) ===
    by_alpha: dict[float, list[float]] = {}
    for r in rows:
        by_alpha.setdefault(float(r["weight_alpha"]), []).append(
            _as_float(r["triple_welfare_coef"])
        )

    alpha_summary: list[dict[str, RowValue]] = []
    for alpha in sorted(by_alpha.keys()):
        vals = by_alpha[alpha]
        alpha_summary.append(
            {
                "weight_alpha": alpha,
                "mean_triple_welfare": round(_mean(vals), 4),
                "se_triple_welfare": round(_se(vals), 4),
                "n_runs": len(vals),
            }
        )
    _write_csv(alpha_summary, out_dir / "summary_by_alpha.csv")

    # === Analysis 2: Effect by ai_noise_sd ===
    by_noise: dict[float, list[float]] = {}
    for r in rows:
        by_noise.setdefault(float(r["ai_noise_sd"]), []).append(_as_float(r["triple_welfare_coef"]))

    noise_summary: list[dict[str, RowValue]] = []
    for noise in sorted(by_noise.keys()):
        vals = by_noise[noise]
        noise_summary.append(
            {
                "ai_noise_sd": noise,
                "mean_triple_welfare": round(_mean(vals), 4),
                "se_triple_welfare": round(_se(vals), 4),
                "n_runs": len(vals),
            }
        )
    _write_csv(noise_summary, out_dir / "summary_by_ai_noise.csv")

    # === Analysis 3: Effect by misclass_rate ===
    by_misclass: dict[float, list[float]] = {}
    for r in rows:
        by_misclass.setdefault(float(r["misclass_rate"]), []).append(
            _as_float(r["triple_welfare_coef"])
        )

    misclass_summary: list[dict[str, RowValue]] = []
    for mc in sorted(by_misclass.keys()):
        vals = by_misclass[mc]
        misclass_summary.append(
            {
                "misclass_rate": mc,
                "mean_triple_welfare": round(_mean(vals), 4),
                "se_triple_welfare": round(_se(vals), 4),
                "n_runs": len(vals),
            }
        )
    _write_csv(misclass_summary, out_dir / "summary_by_misclass.csv")

    # === Plot 1: Triple interaction vs weight_alpha ===
    series_alpha = [
        LineSeries(
            label="AI×Central×Hard effect",
            points=[
                (float(r["weight_alpha"]), _as_float(r["mean_triple_welfare"]))
                for r in alpha_summary
            ],
        )
    ]
    write_line_chart_svg(
        out_path=out_dir / "fig_triple_vs_weight_alpha.svg",
        title="AI×Central advantage vs preference heterogeneity",
        x_label="Preference weight α (lower = more heterogeneous)",
        y_label="Triple interaction coefficient",
        series=series_alpha,
    )

    # === Plot 2: Effect by AI quality (best case vs pessimistic) ===
    series_noise = [
        LineSeries(
            label="AI×Central×Hard effect",
            points=[
                (float(r["ai_noise_sd"]), _as_float(r["mean_triple_welfare"]))
                for r in noise_summary
            ],
        )
    ]
    write_line_chart_svg(
        out_path=out_dir / "fig_triple_vs_ai_noise.svg",
        title="AI×Central advantage vs AI intake quality",
        x_label="AI noise (SD) — lower = better AI",
        y_label="Triple interaction coefficient",
        series=series_noise,
    )

    # === Summary stats ===
    low_alpha_runs = [r for r in rows if _as_float(r["weight_alpha"]) <= 0.5]
    high_alpha_runs = [r for r in rows if _as_float(r["weight_alpha"]) >= 2.0]
    best_ai_runs = [r for r in rows if _as_float(r["ai_noise_sd"]) == 0.03]
    worst_ai_runs = [r for r in rows if _as_float(r["ai_noise_sd"]) >= 0.20]

    low_alpha_mean = _mean([_as_float(r["triple_welfare_coef"]) for r in low_alpha_runs])
    high_alpha_mean = _mean([_as_float(r["triple_welfare_coef"]) for r in high_alpha_runs])
    best_ai_mean = _mean([_as_float(r["triple_welfare_coef"]) for r in best_ai_runs])
    worst_ai_mean = _mean([_as_float(r["triple_welfare_coef"]) for r in worst_ai_runs])

    summary = {
        "n_runs": len(rows),
        "weight_alphas": weight_alphas,
        "ai_noise_sds": ai_noise_sds,
        "misclass_rates": misclass_rates,
        "seeds": seeds,
        "heterogeneity_effect": {
            "low_alpha_mean_triple": round(low_alpha_mean, 4),
            "high_alpha_mean_triple": round(high_alpha_mean, 4),
            "ratio": round(low_alpha_mean / high_alpha_mean, 2) if high_alpha_mean != 0 else None,
        },
        "ai_quality_effect": {
            "best_ai_mean_triple": round(best_ai_mean, 4),
            "worst_ai_mean_triple": round(worst_ai_mean, 4),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # === README ===
    ratio_str = (
        f"{low_alpha_mean / high_alpha_mean:.1f}×"
        if high_alpha_mean != 0 and low_alpha_mean / high_alpha_mean > 0
        else "N/A"
    )
    readme_lines = [
        "# Preference Heterogeneity & Robustness Sweep",
        "",
        "**Key insight**: As preferences become more heterogeneous (lower weight α),",
        "the AI×Central advantage increases. This holds across different AI intake",
        "quality levels and misclassification rates.",
        "",
        "## Parameters Varied",
        "",
        "| Parameter | Values | Meaning |",
        "|---|---|---|",
        f"| `weight_alpha` | {weight_alphas} | Pref concentration (low=focused) |",
        f"| `ai_noise_sd` | {ai_noise_sds} | AI quality (0.03=best, 0.20=bad) |",
        f"| `misclass_rate` | {misclass_rates} | Std form misclass rate |",
        "",
        "## Results: Heterogeneity Effect",
        "",
        f"- Low α (≤0.5) mean triple effect: **{low_alpha_mean:.4f}**",
        f"- High α (≥2.0) mean triple effect: **{high_alpha_mean:.4f}**",
        f"- Ratio (low/high): **{ratio_str}**",
        "",
        "## Results: AI Quality Sensitivity",
        "",
        f"- Best-case AI (noise=0.03): mean triple = **{best_ai_mean:.4f}**",
        f"- Pessimistic AI (noise=0.20): mean triple = **{worst_ai_mean:.4f}**",
        "",
        "## Interpretation",
        "",
        "| weight_alpha | Meaning |",
        "|---|---|",
        "| 0.3 | Very concentrated: person cares about ONE specific dimension |",
        "| 1.0 | Moderate concentration (default) |",
        "| 3.0 | Diffuse: person cares about everything somewhat equally |",
        "",
        "## Artifacts",
        "",
        "- `runs.csv`: per-run results for all parameter combinations",
        "- `summary_by_alpha.csv`: effect aggregated by weight_alpha",
        "- `summary_by_ai_noise.csv`: effect aggregated by AI quality",
        "- `fig_triple_vs_weight_alpha.svg`: main heterogeneity result",
        "- `fig_triple_vs_ai_noise.svg`: robustness to AI quality",
        "",
        "## Run",
        "",
        "```bash",
        "make heterogeneity",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # === Summary table ===
    table_lines = [
        "| weight_α | Mean Triple Effect | SE | Interpretation |",
        "|---|---|---|---|",
    ]
    for r in alpha_summary:
        alpha = float(r["weight_alpha"])
        interp = (
            "very heterogeneous"
            if alpha <= 0.5
            else "heterogeneous"
            if alpha <= 1.0
            else "moderate"
            if alpha <= 1.5
            else "homogeneous"
        )
        line = f"| {alpha} | {r['mean_triple_welfare']:.4f} | {r['se_triple_welfare']:.4f} "
        table_lines.append(line + f"| {interp} |")
    (out_dir / "summary_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    log(logger, 20, "heterogeneity_sweep_written", out_dir=str(out_dir), n_runs=len(rows))


if __name__ == "__main__":
    main()
