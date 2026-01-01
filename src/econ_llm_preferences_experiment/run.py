from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from econ_llm_preferences_experiment.analysis import Metrics, compute_metrics
from econ_llm_preferences_experiment.elicitation import (
    ai_conversation_transcript,
    parse_batch_with_gpt,
    standard_form_text,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    DeferredAcceptanceParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
    deferred_acceptance,
)
from econ_llm_preferences_experiment.models import DIMENSIONS, Category
from econ_llm_preferences_experiment.openai_client import OpenAIClient
from econ_llm_preferences_experiment.plotting import Bar, write_bar_chart_svg
from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    inferred_value_matrix,
    preference_density_proxy,
)

logger = get_logger(__name__)

CentralMechanism = Literal["recs", "da"]


def _write_markdown_table(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _arm_label(elicitation: str, mechanism: str) -> str:
    return f"{elicitation}_{mechanism}"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {type(value).__name__}")


def run_once(
    *,
    category: Category,
    client: OpenAIClient,
    market_params: MarketParams,
    replications: int,
    seed: int,
    attention_cost: float,
    central_mechanism: CentralMechanism = "recs",
    da_proposer_side: Literal["customer", "provider"] = "customer",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)  # nosec B311
    customers, providers = generate_population(
        rng=rng,
        category=category,
        n_customers=market_params.n_customers,
        n_providers=market_params.n_providers,
    )

    truth = {a.agent_id: a for a in customers + providers}
    std_texts = {a.agent_id: standard_form_text(a) for a in customers + providers}
    ai_texts = {a.agent_id: ai_conversation_transcript(a, rng=rng) for a in customers + providers}

    log(logger, 20, "llm_parsing_start", category=category, agents=len(truth))
    parsed_std = parse_batch_with_gpt(
        client=client, texts_by_agent_id=std_texts, truth_by_agent_id=truth
    )
    parsed_ai = parse_batch_with_gpt(
        client=client, texts_by_agent_id=ai_texts, truth_by_agent_id=truth
    )
    log(logger, 20, "llm_parsing_done", category=category)

    inferred_std = {a.agent_id: a for a in parsed_std.inferred}
    inferred_ai = {a.agent_id: a for a in parsed_ai.inferred}

    def weights(side: str, which: str) -> tuple[tuple[float, ...], ...]:
        src = inferred_std if which == "standard" else inferred_ai
        agents = customers if side == "customer" else providers
        return tuple(src[a.agent_id].weights for a in agents)

    rows: list[dict[str, object]] = []
    meta: dict[str, object] = {
        "category": category,
        "seed": seed,
        "replications": replications,
        "n_customers": market_params.n_customers,
        "n_providers": market_params.n_providers,
        "dimensions": list(DIMENSIONS),
        "accept_threshold": market_params.accept_threshold,
        "epsilon": market_params.epsilon,
        "attention_cost": attention_cost,
        "central_mechanism": central_mechanism,
        "da_proposer_side": da_proposer_side,
    }

    arm_metrics: dict[str, list[Metrics]] = {
        _arm_label("standard", "search"): [],
        _arm_label("standard", "central"): [],
        _arm_label("ai", "search"): [],
        _arm_label("ai", "central"): [],
    }
    d_hat_i: dict[str, list[float]] = {"standard": [], "ai": []}
    d_hat_j: dict[str, list[float]] = {"standard": [], "ai": []}

    total_value_per_customer: dict[str, list[float]] = {k: [] for k in arm_metrics}
    attention_per_customer: dict[str, list[float]] = {k: [] for k in arm_metrics}
    net_welfare_per_customer: dict[str, list[float]] = {k: [] for k in arm_metrics}

    weights_customer = {
        "standard": weights("customer", "standard"),
        "ai": weights("customer", "ai"),
    }
    weights_provider = {
        "standard": weights("provider", "standard"),
        "ai": weights("provider", "ai"),
    }

    n_customers = len(customers)

    for r in range(replications):
        rep_rng = random.Random(seed * 10_000 + r)  # nosec B311
        market = generate_market_instance(
            rng=rep_rng,
            customers=customers,
            providers=providers,
            idiosyncratic_noise_sd=market_params.idiosyncratic_noise_sd,
        )

        for elicitation in ("standard", "ai"):
            vhat_c = inferred_value_matrix(
                weights_by_agent=weights_customer[elicitation],
                partner_attributes=market.provider_attributes,
            )
            vhat_p = inferred_value_matrix(
                weights_by_agent=weights_provider[elicitation],
                partner_attributes=market.customer_attributes,
            )
            d_hat_i[elicitation].append(
                preference_density_proxy(
                    v_true=market.v_customer, v_hat=vhat_c, epsilon=market_params.epsilon
                )
            )
            d_hat_j[elicitation].append(
                preference_density_proxy(
                    v_true=market.v_provider, v_hat=vhat_p, epsilon=market_params.epsilon
                )
            )

            outcome_search = decentralized_search(
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
                v_customer_hat=vhat_c,
                accept_threshold=market_params.accept_threshold,
                params=SearchParams(max_rounds=30),
            )
            metrics_search = compute_metrics(
                outcome=outcome_search,
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
            )
            arm_search = _arm_label(elicitation, "search")
            arm_metrics[arm_search].append(metrics_search)

            tv_sum = sum(
                market.v_customer[i][j] + market.v_provider[j][i] for i, j in outcome_search.matches
            )
            attn_total = outcome_search.proposals + outcome_search.accept_decisions
            total_value_per_customer[arm_search].append(tv_sum / n_customers)
            attention_per_customer[arm_search].append(attn_total / n_customers)
            net_welfare_per_customer[arm_search].append(
                (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
            )

            if central_mechanism == "recs":
                outcome_central = centralized_recommendations(
                    v_customer_true=market.v_customer,
                    v_provider_true=market.v_provider,
                    v_customer_hat=vhat_c,
                    v_provider_hat=vhat_p,
                    accept_threshold=market_params.accept_threshold,
                    params=CentralizedParams(rec_k=3),
                )
            elif central_mechanism == "da":
                outcome_central = deferred_acceptance(
                    v_customer_true=market.v_customer,
                    v_provider_true=market.v_provider,
                    v_customer_hat=vhat_c,
                    v_provider_hat=vhat_p,
                    accept_threshold=market_params.accept_threshold,
                    params=DeferredAcceptanceParams(proposer_side=da_proposer_side),
                )
            else:
                raise ValueError(f"Unknown central_mechanism: {central_mechanism}")
            metrics_central = compute_metrics(
                outcome=outcome_central,
                v_customer_true=market.v_customer,
                v_provider_true=market.v_provider,
            )
            arm_central = _arm_label(elicitation, "central")
            arm_metrics[arm_central].append(metrics_central)

            tv_sum = sum(
                market.v_customer[i][j] + market.v_provider[j][i]
                for i, j in outcome_central.matches
            )
            attn_total = outcome_central.proposals + outcome_central.accept_decisions
            total_value_per_customer[arm_central].append(tv_sum / n_customers)
            attention_per_customer[arm_central].append(attn_total / n_customers)
            net_welfare_per_customer[arm_central].append(
                (tv_sum / n_customers) - attention_cost * (attn_total / n_customers)
            )

    for elicitation in ("standard", "ai"):
        for mechanism in ("search", "central"):
            arm = _arm_label(elicitation, mechanism)
            metrics = arm_metrics[arm]
            d_i = d_hat_i[elicitation]
            d_j = d_hat_j[elicitation]

            rows.append(
                {
                    "category": category,
                    "arm": arm,
                    "elicitation": elicitation,
                    "mechanism": mechanism,
                    "d_hat_I": round(_mean(d_i), 3),
                    "d_hat_I_se": round(_se(d_i), 3),
                    "d_hat_J": round(_mean(d_j), 3),
                    "d_hat_J_se": round(_se(d_j), 3),
                    "match_rate": round(_mean([m.match_rate for m in metrics]), 3),
                    "match_rate_se": round(_se([m.match_rate for m in metrics]), 3),
                    "mean_total_value": round(_mean([m.mean_total_value for m in metrics]), 3),
                    "mean_total_value_se": round(_se([m.mean_total_value for m in metrics]), 3),
                    "mean_rounds": round(_mean([m.mean_rounds for m in metrics]), 3),
                    "proposals_per_match": round(
                        _mean([m.proposals_per_match for m in metrics]), 3
                    ),
                    "accept_decisions_per_match": round(
                        _mean([m.accept_decisions_per_match for m in metrics]), 3
                    ),
                    "attention_per_match": round(
                        _mean(
                            [m.proposals_per_match + m.accept_decisions_per_match for m in metrics]
                        ),
                        3,
                    ),
                    "total_value_per_customer": round(_mean(total_value_per_customer[arm]), 3),
                    "total_value_per_customer_se": round(_se(total_value_per_customer[arm]), 3),
                    "attention_per_customer": round(_mean(attention_per_customer[arm]), 3),
                    "attention_per_customer_se": round(_se(attention_per_customer[arm]), 3),
                    "net_welfare_per_customer": round(_mean(net_welfare_per_customer[arm]), 3),
                    "net_welfare_per_customer_se": round(_se(net_welfare_per_customer[arm]), 3),
                }
            )

    def series_match_rate(arm: str) -> list[float]:
        return [m.match_rate for m in arm_metrics[arm]]

    def series_net_welfare(arm: str) -> list[float]:
        return net_welfare_per_customer[arm]

    did_match_rate = [
        (series_match_rate("ai_central")[t] - series_match_rate("standard_central")[t])
        - (series_match_rate("ai_search")[t] - series_match_rate("standard_search")[t])
        for t in range(replications)
    ]
    did_net_welfare = [
        (series_net_welfare("ai_central")[t] - series_net_welfare("standard_central")[t])
        - (series_net_welfare("ai_search")[t] - series_net_welfare("standard_search")[t])
        for t in range(replications)
    ]
    d_hat_i_diff = [d_hat_i["ai"][t] - d_hat_i["standard"][t] for t in range(replications)]
    d_hat_j_diff = [d_hat_j["ai"][t] - d_hat_j["standard"][t] for t in range(replications)]

    meta["effects"] = {
        "category": category,
        "d_hat_I_ai_minus_standard": _mean(d_hat_i_diff),
        "d_hat_I_ai_minus_standard_se": _se(d_hat_i_diff),
        "d_hat_J_ai_minus_standard": _mean(d_hat_j_diff),
        "d_hat_J_ai_minus_standard_se": _se(d_hat_j_diff),
        "match_rate_did": _mean(did_match_rate),
        "match_rate_did_se": _se(did_match_rate),
        "net_welfare_did": _mean(did_net_welfare),
        "net_welfare_did_se": _se(did_net_welfare),
    }

    return rows, meta


def _write_report(
    *,
    out_dir: Path,
    summary_rows: list[dict[str, object]],
    effects_rows: list[dict[str, object]] | None,
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = out_dir / "summary_table.md"
    _write_markdown_table(summary_rows, md_path)

    if effects_rows:
        effects_csv = out_dir / "effects_table.csv"
        with effects_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(effects_rows[0].keys()))
            writer.writeheader()
            writer.writerows(effects_rows)
        effects_md = out_dir / "effects_table.md"
        _write_markdown_table(effects_rows, effects_md)

    by_cat: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_cat.setdefault(str(row["category"]), []).append(row)
    for category, rows in by_cat.items():
        rows_sorted = sorted(rows, key=lambda r: str(r["arm"]))
        labels = [str(r["arm"]) for r in rows_sorted]

        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_match_rate.svg",
            title=f"Match rate by arm ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["match_rate"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="match_rate",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_total_value.svg",
            title=f"Mean total value by arm ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["mean_total_value"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="mean_total_value",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_d_hat_I.svg",
            title=f"Customer preference density proxy d̂_I ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["d_hat_I"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="d_hat_I",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_d_hat_J.svg",
            title=f"Provider preference density proxy d̂_J ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["d_hat_J"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="d_hat_J",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_attention_per_match.svg",
            title=f"Attention actions per match ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["attention_per_match"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="(proposals + accept decisions) / match",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_net_welfare_per_customer.svg",
            title=f"Net welfare per customer ({category})",
            bars=[
                Bar(label=label, value=_as_float(row["net_welfare_per_customer"]))
                for label, row in zip(labels, rows_sorted, strict=True)
            ],
            y_label="value_per_customer - λ·attention_per_customer",
        )

    def metric(category: str, arm: str, key: str) -> float:
        for row in summary_rows:
            if str(row["category"]) == category and str(row["arm"]) == arm:
                return _as_float(row[key])
        raise KeyError((category, arm, key))

    interaction: dict[str, dict[str, float]] = {}
    for category in by_cat:
        a = metric(category, "standard_search", "match_rate")
        b = metric(category, "ai_search", "match_rate")
        c = metric(category, "standard_central", "match_rate")
        d = metric(category, "ai_central", "match_rate")
        interaction[category] = {
            "match_rate_interaction": (d - c) - (b - a),
            "central_minus_search_standard": c - a,
            "central_minus_search_ai": d - b,
        }
    metadata["interaction"] = interaction

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    diagnostic_lines: list[str] = []
    lambda_lines: list[str] = []
    for category in sorted(by_cat.keys()):
        d_i_std = metric(category, "standard_search", "d_hat_I")
        d_j_std = metric(category, "standard_search", "d_hat_J")
        d_i_ai = metric(category, "ai_search", "d_hat_I")
        d_j_ai = metric(category, "ai_search", "d_hat_J")
        m_std = metric(category, "standard_central", "match_rate")
        m_ai = metric(category, "ai_central", "match_rate")
        did = interaction[category]["match_rate_interaction"]
        nw_s_std = metric(category, "standard_search", "net_welfare_per_customer")
        nw_c_std = metric(category, "standard_central", "net_welfare_per_customer")
        nw_s_ai = metric(category, "ai_search", "net_welfare_per_customer")
        nw_c_ai = metric(category, "ai_central", "net_welfare_per_customer")
        nw_did = (nw_c_ai - nw_c_std) - (nw_s_ai - nw_s_std)
        diagnostic_lines.extend(
            [
                f"- {category}: d̂_I {d_i_std:.3f} → {d_i_ai:.3f}; d̂_J {d_j_std:.3f} → {d_j_ai:.3f}",
                f"  central match_rate {m_std:.3f} → {m_ai:.3f}; DiD {did:+.3f}",
                (
                    f"  net_welfare/customer (central - search): "
                    f"standard {(nw_c_std - nw_s_std):+.3f}; "
                    f"ai {(nw_c_ai - nw_s_ai):+.3f}; "
                    f"DiD {nw_did:+.3f}"
                ),
            ]
        )

        v_s_std = metric(category, "standard_search", "total_value_per_customer")
        v_c_std = metric(category, "standard_central", "total_value_per_customer")
        a_s_std = metric(category, "standard_search", "attention_per_customer")
        a_c_std = metric(category, "standard_central", "attention_per_customer")
        denom_std = a_s_std - a_c_std
        ls_std = None if denom_std <= 0 else (v_s_std - v_c_std) / denom_std

        v_s_ai = metric(category, "ai_search", "total_value_per_customer")
        v_c_ai = metric(category, "ai_central", "total_value_per_customer")
        a_s_ai = metric(category, "ai_search", "attention_per_customer")
        a_c_ai = metric(category, "ai_central", "attention_per_customer")
        denom_ai = a_s_ai - a_c_ai
        ls_ai = None if denom_ai <= 0 else (v_s_ai - v_c_ai) / denom_ai
        if ls_std is not None and ls_ai is not None:
            lambda_lines.append(f"- {category}: λ* standard={ls_std:.4f}, ai={ls_ai:.4f}")

    central_mechanism: str | None = None
    da_proposer_side: str | None = None

    def _maybe_set_mechanism(meta_obj: object) -> None:
        nonlocal central_mechanism, da_proposer_side
        if not isinstance(meta_obj, dict):
            return
        cm = meta_obj.get("central_mechanism")
        if isinstance(cm, str):
            central_mechanism = cm
        proposer = meta_obj.get("da_proposer_side")
        if isinstance(proposer, str):
            da_proposer_side = proposer

    _maybe_set_mechanism(metadata)
    _maybe_set_mechanism(metadata.get("category_easy"))
    _maybe_set_mechanism(metadata.get("category_hard"))

    if central_mechanism == "da":
        proposer = f", proposer={da_proposer_side}" if da_proposer_side else ""
        mechanism_line = f"Mechanism: central = deferred acceptance (stable matching){proposer}."
        so_what_lines = [
            "So what: Roth-style benchmark (stable matching under hat-based rankings).",
            "We apply a mutual acceptability cutoff, and centralization is very attention-light.",
            "Focus on the net-welfare gap and the ROI boundary (λ*).",
        ]
    else:
        mechanism_line = "Mechanism: central = top‑k recommendations + max-cardinality matching."
        so_what_lines = [
            "So what: AI elicitation raises the preference-density proxies (d̂_I, d̂_J),",
            "and disproportionately improves centralized recommendations in",
            "low-describability categories.",
        ]

    readme = "\n".join(
        [
            "# Latest run",
            "",
            mechanism_line,
            "",
            *so_what_lines,
            "",
            "Key diagnostics (match_rate DiD):",
            *diagnostic_lines,
            "",
            "ROI boundary (λ*): central beats search if λ > λ*.",
            *lambda_lines,
            "",
            "Artifacts:",
            "- `summary_table.csv` / `summary_table.md`: arm-by-arm outcomes",
            "- `effects_table.csv` / `effects_table.md`: key contrasts (DiD, etc.)",
            "- `fig_*`: quick plots by category",
            "- `run_metadata.json`: parameters + seeds",
            "",
            "Optional regime map:",
            "- Run `make regime` to write:",
            "  - `regime_map_hard_net_welfare_diff.svg` (at the configured attention_cost)",
            "  - `regime_map_hard_lambda_star.svg` (ROI boundary λ*; central wins if λ > λ*)",
            "  - `regime_grid_hard.csv` (underlying grid)",
            "",
            "Interpretation tip: look for the interaction—`ai_central` beats `standard_central`",
            "by more than `ai_search` beats `standard_search`.",
            "",
            "Figures (easy):",
            "- `fig_easy_match_rate.svg`",
            "- `fig_easy_total_value.svg`",
            "- `fig_easy_d_hat_I.svg`",
            "- `fig_easy_d_hat_J.svg`",
            "- `fig_easy_attention_per_match.svg`",
            "- `fig_easy_net_welfare_per_customer.svg`",
            "",
            "Figures (hard):",
            "- `fig_hard_match_rate.svg`",
            "- `fig_hard_total_value.svg`",
            "- `fig_hard_d_hat_I.svg`",
            "- `fig_hard_d_hat_J.svg`",
            "- `fig_hard_attention_per_match.svg`",
            "- `fig_hard_net_welfare_per_customer.svg`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/latest")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-cost", type=float, default=0.01)
    parser.add_argument(
        "--central-mechanism",
        choices=["recs", "da"],
        default="recs",
        help="Central mechanism: 'recs' (top-k recs + max matching) or 'da' (deferred acceptance).",
    )
    parser.add_argument(
        "--da-proposer-side",
        choices=["customer", "provider"],
        default="customer",
        help="Only used when --central-mechanism=da.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    market_params = MarketParams()
    client = OpenAIClient(max_calls=9)

    all_rows: list[dict[str, object]] = []
    meta: dict[str, object] = {
        "replications": args.replications,
        "seed": args.seed,
        "market_params": asdict(market_params),
        "attention_cost": args.attention_cost,
        "model": client.env.model,
        "base_url": client.env.base_url,
    }

    effects_rows: list[dict[str, object]] = []

    for category in ("easy", "hard"):
        rows, cat_meta = run_once(
            category=category,
            client=client,
            market_params=market_params,
            replications=args.replications,
            seed=args.seed,
            attention_cost=args.attention_cost,
            central_mechanism=args.central_mechanism,
            da_proposer_side=args.da_proposer_side,
        )
        all_rows.extend(rows)
        meta[f"category_{category}"] = cat_meta

        effects_obj = cat_meta.get("effects")
        if isinstance(effects_obj, dict):
            effects: dict[str, object] = effects_obj
            row_by_arm = {str(row["arm"]): row for row in rows}
            std_search = row_by_arm["standard_search"]
            std_central = row_by_arm["standard_central"]
            ai_search = row_by_arm["ai_search"]
            ai_central = row_by_arm["ai_central"]

            def _lambda_star(search: dict[str, object], central: dict[str, object]) -> float | None:
                v_s = _as_float(search["total_value_per_customer"])
                v_c = _as_float(central["total_value_per_customer"])
                a_s = _as_float(search["attention_per_customer"])
                a_c = _as_float(central["attention_per_customer"])
                denom = a_s - a_c
                if denom <= 0:
                    return None
                return (v_s - v_c) / denom

            ls_standard = _lambda_star(std_search, std_central)
            ls_ai = _lambda_star(ai_search, ai_central)

            effects_rows.append(
                {
                    "category": effects.get("category", category),
                    "d_hat_I_ai_minus_standard": round(
                        _as_float(effects["d_hat_I_ai_minus_standard"]), 3
                    ),
                    "d_hat_I_ai_minus_standard_se": round(
                        _as_float(effects["d_hat_I_ai_minus_standard_se"]), 3
                    ),
                    "d_hat_J_ai_minus_standard": round(
                        _as_float(effects["d_hat_J_ai_minus_standard"]), 3
                    ),
                    "d_hat_J_ai_minus_standard_se": round(
                        _as_float(effects["d_hat_J_ai_minus_standard_se"]), 3
                    ),
                    "match_rate_did": round(_as_float(effects["match_rate_did"]), 3),
                    "match_rate_did_se": round(_as_float(effects["match_rate_did_se"]), 3),
                    "net_welfare_did": round(_as_float(effects["net_welfare_did"]), 3),
                    "net_welfare_did_se": round(_as_float(effects["net_welfare_did_se"]), 3),
                    "lambda_star_standard": None if ls_standard is None else round(ls_standard, 4),
                    "lambda_star_ai": None if ls_ai is None else round(ls_ai, 4),
                }
            )

    _write_report(out_dir=out_dir, summary_rows=all_rows, effects_rows=effects_rows, metadata=meta)
    log(logger, 20, "report_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
