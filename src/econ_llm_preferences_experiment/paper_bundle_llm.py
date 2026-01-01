"""
Paper bundle using ONLY real LLM-backed experiments.

NO synthetic simulations - all results use real GPT-5.2 calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, TypeAlias

from econ_llm_preferences_experiment.logging_utils import get_logger, log

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md_table_lines(rows: list[dict[str, RowValue]]) -> list[str]:
    if not rows:
        return [""]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper bundle using ONLY real LLM-backed experiments (no synthetic)"
    )
    parser.add_argument("--out", default="reports/paper_llm_latest")
    parser.add_argument("--main-dir", default="reports/latest")
    parser.add_argument("--ablations-dir", default="reports/ablations_latest")
    parser.add_argument("--congestion-dir", default="reports/latest_congestion")
    parser.add_argument("--heterogeneity-dir", default="reports/heterogeneity_llm_latest")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_exp = Path(args.main_dir)
    ablations = Path(args.ablations_dir)
    congestion = Path(args.congestion_dir)
    heterogeneity = Path(args.heterogeneity_dir)

    key_rows: list[dict[str, RowValue]] = []

    # === 1. Main 2×2 experiment (real LLM via parse_batch_with_gpt) ===
    effects_rows = _read_csv(main_exp / "effects_table.csv")
    for row in effects_rows:
        key_rows.append(
            {
                "block": "Main_2x2_LLM",
                "metric": "d_hat_I improvement (AI vs standard)",
                "category": row["category"],
                "estimate": round(_as_float(row["d_hat_I_ai_minus_standard"]), 4),
                "se": round(_as_float(row["d_hat_I_ai_minus_standard_se"]), 4),
                "notes": "real GPT-5.2 parsing",
            }
        )
        key_rows.append(
            {
                "block": "Main_2x2_LLM",
                "metric": "match_rate DiD",
                "category": row["category"],
                "estimate": round(_as_float(row["match_rate_did"]), 4),
                "se": round(_as_float(row["match_rate_did_se"]), 4),
                "notes": "(AI_central - AI_search) - (std_central - std_search)",
            }
        )
        key_rows.append(
            {
                "block": "Main_2x2_LLM",
                "metric": "net_welfare DiD",
                "category": row["category"],
                "estimate": round(_as_float(row["net_welfare_did"]), 4),
                "se": round(_as_float(row["net_welfare_did_se"]), 4),
                "notes": "welfare impact of AI+Central interaction",
            }
        )
        key_rows.append(
            {
                "block": "Main_2x2_LLM",
                "metric": "λ* (ROI threshold)",
                "category": row["category"],
                "estimate": round(_as_float(row["lambda_star_ai"]), 4),
                "se": "",
                "notes": "central beats search if attention_cost > λ*",
            }
        )

    # === 2. LLM ablations (real GPT-5.2) ===
    for category in ("easy", "hard"):
        pq_rows = _read_csv(ablations / f"parsing_quality_{category}.csv")
        for mode in ("form_top3", "free_text_gpt", "chat_gpt"):
            for side in ("customer", "provider"):
                row = next(
                    r
                    for r in pq_rows
                    if r["mode"] == mode and r["side"] == side and r["category"] == category
                )
                key_rows.append(
                    {
                        "block": "LLM_ablations",
                        "metric": f"{mode} parsing quality",
                        "category": f"{category}_{side}",
                        "estimate": round(_as_float(row["mean_l1"]), 4),
                        "se": "",
                        "notes": f"top1_acc={float(row['top1_accuracy']):.3f}",
                    }
                )

    # === 3. Congestion experiment (real GPT-5.2) ===
    for category in ("easy", "hard"):
        sat_csv = congestion / f"congestion_saturation_{category}.csv"
        sat_rows = _read_csv(sat_csv)

        r0 = next(r for r in sat_rows if abs(_as_float(r["saturation"]) - 0.0) <= 1e-9)
        r100 = next(r for r in sat_rows if abs(_as_float(r["saturation"]) - 1.0) <= 1e-9)
        key_rows.append(
            {
                "block": "Congestion_LLM",
                "metric": "Δ net_welfare (100% - 0%)",
                "category": category,
                "estimate": round(
                    _as_float(r100["net_welfare_per_customer"])
                    - _as_float(r0["net_welfare_per_customer"]),
                    4,
                ),
                "se": "",
                "notes": f"inbox@100%={float(r100['inbox_per_provider_per_day']):.1f}/day",
            }
        )

    # === 4. Heterogeneity LLM sweep (real GPT-5.2) ===
    het_rows = _read_csv(heterogeneity / "runs.csv")
    for row in het_rows:
        if row["category"] == "hard":  # Focus on hard category
            key_rows.append(
                {
                    "block": "Heterogeneity_LLM",
                    "metric": "utility_noise effect on DiD",
                    "category": f"utility_noise={row['utility_noise_sd']}",
                    "estimate": round(_as_float(row["net_welfare_did"]), 4),
                    "se": round(_as_float(row["net_welfare_did_se"]), 4),
                    "notes": "real GPT-5.2 parsing",
                }
            )

    _write_csv(key_rows, out_dir / "key_results.csv")

    # Write markdown version
    md_lines = [
        "# Paper Bundle (LLM-Only, No Synthetic)",
        "",
        "**All results use real GPT-5.2 calls** via `parse_batch_with_gpt()`.",
        "",
        "## Summary",
        "",
        "| Experiment | Source | Real LLM? |",
        "|------------|--------|-----------|",
        f"| Main 2×2 | `{main_exp}` | ✅ Yes |",
        f"| Ablations | `{ablations}` | ✅ Yes |",
        f"| Congestion | `{congestion}` | ✅ Yes |",
        f"| Heterogeneity | `{heterogeneity}` | ✅ Yes |",
        "",
        "## Key Results",
        "",
        *_md_table_lines(key_rows),
        "",
        "## How to regenerate",
        "",
        "```bash",
        "make experiment        # Main 2×2 with GPT",
        "make ablations         # Parsing quality with GPT",
        "make congestion        # Saturation effects with GPT",
        "make heterogeneity-llm # Utility noise sweep with GPT",
        "make paper-bundle-llm  # This bundle",
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Summary JSON
    summary = {
        "n_rows": len(key_rows),
        "sources": {
            "main_exp": str(main_exp),
            "ablations": str(ablations),
            "congestion": str(congestion),
            "heterogeneity": str(heterogeneity),
        },
        "llm_model": "gpt-5.2",
        "synthetic_results_included": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log(logger, 20, "paper_bundle_llm_written", out_dir=str(out_dir), n_rows=len(key_rows))


if __name__ == "__main__":
    main()
