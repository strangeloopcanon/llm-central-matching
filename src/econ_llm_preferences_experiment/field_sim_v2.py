from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from econ_llm_preferences_experiment.econometrics import OLSResult, ols_cluster_robust
from econ_llm_preferences_experiment.home_services import (
    HomeServicesTask,
    sample_task,
)
from econ_llm_preferences_experiment.home_services import (
    draw_budget as _draw_budget_for_task,
)
from econ_llm_preferences_experiment.home_services import (
    draw_complexity as _draw_complexity_for_task,
)
from econ_llm_preferences_experiment.home_services import (
    draw_weirdness as _draw_weirdness_for_task,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.models import DIMENSIONS, Category
from econ_llm_preferences_experiment.plotting import (
    Bar,
    LineSeries,
    write_bar_chart_svg,
    write_line_chart_svg,
)
from econ_llm_preferences_experiment.simulation import MarketParams

logger = get_logger(__name__)

Elicitation: TypeAlias = Literal["standard", "ai"]
Mechanism: TypeAlias = Literal["search", "central"]

RowValue: TypeAlias = str | float | int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _se(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _dirichlet(rng: random.Random, k: int, *, alpha: float = 1.2) -> tuple[float, ...]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    if total <= 0:
        return tuple(1.0 / k for _ in range(k))
    return tuple(x / total for x in draws)


def _herfindahl(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    shares = [c / total for c in counts]
    return sum(s * s for s in shares)


@dataclass(frozen=True)
class TruthProvider:
    provider_id: str
    city_id: str
    weights: tuple[float, ...]
    attributes: tuple[float, ...]
    cost_base: float
    schedule_slots: frozenset[int]
    licensed: bool
    insured: bool


@dataclass(frozen=True)
class TruthJob:
    job_id: str
    city_id: str
    cell_id: str
    category: Category
    task_id: str
    task_label: str
    weights: tuple[float, ...]
    attributes: tuple[float, ...]
    budget_true: float
    schedule_slots: frozenset[int]
    requires_license: bool
    requires_insurance: bool
    complexity: float
    weirdness: float


@dataclass(frozen=True)
class SpecProvider:
    provider_id: str
    weights_hat: tuple[float, ...]
    cost_base_hat: float
    schedule_slots_hat: frozenset[int]
    licensed_hat: bool
    insured_hat: bool


@dataclass(frozen=True)
class SpecJob:
    job_id: str
    weights_hat: tuple[float, ...]
    budget_reported: float
    schedule_slots_reported: frozenset[int] | None
    requires_license_reported: bool | None
    requires_insurance_reported: bool | None
    complexity_reported: float
    weirdness_reported: float
    edited: bool
    strategic_shade: bool


@dataclass(frozen=True)
class FieldV2Params:
    horizon_days: int = 7
    slots_per_day: int = 3
    provider_weekly_capacity: int = 5
    provider_daily_screen_cap: int = 8
    search_k_per_day: int = 2
    search_explore_share: float = 0.35
    search_reveal_rate: float = 0.18
    central_rec_k: int = 5
    central_days: tuple[int, ...] = (1,)
    accept_threshold: float = 0.25
    idiosyncratic_noise_sd: float = 0.08
    epsilon: float = 0.10
    base_markup: float = 0.22
    demand_markup: float = 0.18
    budget_markup: float = 0.10
    attention_cost: float = 0.01
    cancel_base: float = 0.05
    cancel_price_reneg: float = 0.15
    cancel_weirdness_reveal: float = 0.55
    ai_intake_reveal: float = 0.40
    compliance_ai: float = 0.85
    contamination_ai_in_control: float = 0.05
    compliance_central: float = 0.90
    strategic_share: float = 0.35
    shade_frac: float = 0.15
    ai_shade_uplift: float = 0.05
    ai_budget_noise_sd: float = 0.10
    ai_requirement_fn: float = 0.08
    ai_requirement_fp: float = 0.03
    ai_schedule_drop: float = 0.10
    ai_schedule_add: float = 0.05
    ai_edit_big_budget: float = 0.75
    ai_edit_requirement: float = 0.70
    ai_edit_schedule: float = 0.55
    std_weight_misclass_hard: float = 0.70
    std_provider_weight_misclass: float = 0.45
    value_scale_easy: float = 500.0
    value_scale_hard: float = 1100.0
    # Dirichlet alpha for attribute heterogeneity: lower = more concentrated attributes
    dirichlet_alpha: float = 1.2
    # Dirichlet alpha for preference weight heterogeneity:
    # lower = more concentrated (cares about ONE thing)
    weight_alpha: float = 1.0
    # AI intake noise (SD): lower = better AI elicitation, 0.03 is best-case
    ai_weight_noise_sd: float = 0.03


@dataclass(frozen=True)
class CityWeekAssignment:
    city_id: str
    week: int
    cell_easy: tuple[Elicitation, Mechanism]
    cell_hard: tuple[Elicitation, Mechanism]


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    city_id: str
    week: int
    category: Category
    elicitation: Elicitation
    mechanism: Mechanism
    n_jobs: int
    match_rate: float
    mean_days_to_match: float
    cancel_rate: float
    messages_per_job: float
    provider_inbox_per_day: float
    provider_response_rate: float
    avg_price: float
    price_sd: float
    consumer_surplus_per_job: float
    provider_profit_per_job: float
    total_surplus_per_job: float
    net_welfare_per_job: float
    herfindahl_providers: float
    reciprocity_obs: float
    pref_density_i: float
    pref_density_j: float
    pref_density_both: float


def _arm_label(elicitation: Elicitation, mechanism: Mechanism) -> str:
    return f"{elicitation}_{mechanism}"


def _params_with_overrides(base: FieldV2Params, overrides: dict[str, Any]) -> FieldV2Params:
    data = asdict(base)
    for key, value in overrides.items():
        if key not in data:
            raise KeyError(f"Unknown FieldV2Params key: {key}")
        if key == "central_days" and isinstance(value, list):
            data[key] = tuple(int(x) for x in value)
            continue
        data[key] = value

    int_fields = (
        "horizon_days",
        "slots_per_day",
        "provider_weekly_capacity",
        "provider_daily_screen_cap",
        "search_k_per_day",
        "central_rec_k",
    )
    float_fields = (
        "search_explore_share",
        "search_reveal_rate",
        "accept_threshold",
        "idiosyncratic_noise_sd",
        "epsilon",
        "base_markup",
        "demand_markup",
        "budget_markup",
        "attention_cost",
        "cancel_base",
        "cancel_price_reneg",
        "cancel_weirdness_reveal",
        "ai_intake_reveal",
        "compliance_ai",
        "contamination_ai_in_control",
        "compliance_central",
        "strategic_share",
        "shade_frac",
        "ai_shade_uplift",
        "ai_budget_noise_sd",
        "ai_requirement_fn",
        "ai_requirement_fp",
        "ai_schedule_drop",
        "ai_schedule_add",
        "ai_edit_big_budget",
        "ai_edit_requirement",
        "ai_edit_schedule",
        "std_weight_misclass_hard",
        "std_provider_weight_misclass",
        "value_scale_easy",
        "value_scale_hard",
        "dirichlet_alpha",
        "weight_alpha",
        "ai_weight_noise_sd",
    )
    for k in int_fields:
        data[k] = int(data[k])
    for k in float_fields:
        data[k] = float(data[k])
    data["central_days"] = tuple(int(x) for x in data["central_days"])
    return FieldV2Params(**data)


def _weights_standard_from_truth(weights: tuple[float, ...]) -> tuple[float, ...]:
    k = len(weights)
    top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[: min(3, k)]
    # A realistic baseline form tends to elicit a ranked list with coarse intensities.
    # Use a fixed (high/medium/low) pattern to preserve salience of the top choice.
    pattern = (0.60, 0.25, 0.15)
    out = [0.0 for _ in weights]
    for rank, idx in enumerate(top):
        out[idx] = pattern[rank] if rank < len(pattern) else 0.0
    total = sum(out)
    if total <= 0.0:
        return tuple(1.0 / k for _ in out)
    return tuple(x / total for x in out)


def _weights_ai_from_truth(
    *, rng: random.Random, weights: tuple[float, ...], noise_sd: float = 0.03
) -> tuple[float, ...]:
    # AI intake elicits a richer preference profile: close to truth with configurable noise.
    # Lower noise_sd = better AI elicitation (0.03 is best-case, 0.15 is pessimistic).
    noisy = [max(0.0, w + rng.gauss(0.0, noise_sd)) for w in weights]
    s = sum(noisy)
    if s <= 0.0:
        return tuple(1.0 / len(noisy) for _ in noisy)
    return tuple(x / s for x in noisy)


def _draw_weights_dirichlet(
    rng: random.Random, *, k: int, alpha: float, category: Category
) -> tuple[float, ...]:
    """
    Draw preference weights from Dirichlet(alpha, ..., alpha).
    Lower alpha = more concentrated (person cares about ONE thing).
    Higher alpha = more diffuse (person cares about everything equally).

    For 'easy' categories, we zero out non-top-3 weights (simulating simpler preferences).
    """
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    weights = [1.0 / k for _ in range(k)] if total <= 0 else [x / total for x in draws]

    if category == "easy":
        # Easy categories: only top-3 dimensions matter
        top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[:3]
        weights = [weights[i] if i in top else 0.0 for i in range(k)]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    return tuple(weights)


def _draw_budget(rng: random.Random, *, category: Category) -> float:
    if category == "easy":
        return float(rng.lognormvariate(math.log(250.0), 0.35))
    return float(rng.lognormvariate(math.log(700.0), 0.55))


def _draw_complexity(rng: random.Random, *, category: Category) -> float:
    if category == "easy":
        return float(max(0.7, min(1.5, rng.gauss(1.0, 0.15))))
    return float(max(0.8, min(2.4, rng.gauss(1.25, 0.35))))


def _draw_weirdness(rng: random.Random, *, category: Category, complexity: float) -> float:
    if category == "easy":
        base = 0.18 + 0.22 * (complexity - 1.0)
        return _clip01(base + rng.gauss(0.0, 0.12))
    base = 0.35 + 0.30 * (complexity - 1.0)
    return _clip01(base + rng.gauss(0.0, 0.18))


def _sample_task(rng: random.Random, *, category: Category) -> HomeServicesTask:
    return sample_task(rng=rng, category=category)


def _draw_schedule(
    rng: random.Random, *, horizon_days: int, slots_per_day: int, mean_slots: int
) -> frozenset[int]:
    n_slots = horizon_days * slots_per_day
    k = max(1, min(n_slots, int(round(rng.gauss(mean_slots, mean_slots * 0.25)))))
    return frozenset(rng.sample(list(range(n_slots)), k=k))


def _schedule_overlap(a: frozenset[int], b: frozenset[int]) -> bool:
    return bool(a.intersection(b))


def _provider_cost_base(rng: random.Random, provider_attributes: tuple[float, ...]) -> float:
    # Base cost rises with "quality_focus" and "speed_urgency" in the provider profile.
    quality = provider_attributes[1]
    speed = provider_attributes[2]
    price = provider_attributes[0]
    base = 140.0 + 260.0 * quality + 120.0 * speed - 60.0 * price
    base *= float(rng.lognormvariate(0.0, 0.18))
    return max(60.0, base)


def _make_provider_pool(
    *,
    rng: random.Random,
    city_id: str,
    n_providers: int,
    params: FieldV2Params,
) -> tuple[tuple[TruthProvider, ...], dict[str, SpecProvider], dict[str, SpecProvider]]:
    providers: list[TruthProvider] = []
    spec_standard: dict[str, SpecProvider] = {}
    spec_ai: dict[str, SpecProvider] = {}
    n_slots = params.horizon_days * params.slots_per_day

    for j in range(n_providers):
        provider_id = f"{city_id}_p{j:03d}"
        weights = _draw_weights_dirichlet(
            rng, k=len(DIMENSIONS), alpha=params.weight_alpha, category="hard"
        )
        attributes = _dirichlet(rng, len(DIMENSIONS), alpha=params.dirichlet_alpha)
        cost_base = _provider_cost_base(rng, attributes)
        schedule = frozenset(rng.sample(list(range(n_slots)), k=max(2, int(0.65 * n_slots))))
        licensed = rng.random() < 0.72
        insured = rng.random() < 0.68
        tp = TruthProvider(
            provider_id=provider_id,
            city_id=city_id,
            weights=weights,
            attributes=attributes,
            cost_base=cost_base,
            schedule_slots=schedule,
            licensed=licensed,
            insured=insured,
        )
        providers.append(tp)

        weights_hat_std = _weights_standard_from_truth(weights)
        if rng.random() < params.std_provider_weight_misclass:
            k = len(weights)
            top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[: min(3, k)]
            non_top = [idx for idx in range(k) if idx not in top]
            if top and non_top:
                drop = rng.choice(top)
                add = rng.choice(non_top)
                new_top = [idx for idx in top if idx != drop] + [add]
                pattern = (0.60, 0.25, 0.15)
                out = [0.0 for _ in weights]
                for rank, idx in enumerate(new_top[: len(pattern)]):
                    out[idx] = pattern[rank]
                total = sum(out)
                if total > 0.0:
                    weights_hat_std = tuple(x / total for x in out)

        std = SpecProvider(
            provider_id=provider_id,
            weights_hat=weights_hat_std,
            cost_base_hat=cost_base,
            schedule_slots_hat=schedule,
            licensed_hat=licensed,
            insured_hat=insured,
        )
        spec_standard[provider_id] = std

        ai = SpecProvider(
            provider_id=provider_id,
            weights_hat=_weights_ai_from_truth(
                rng=rng, weights=weights, noise_sd=params.ai_weight_noise_sd
            ),
            cost_base_hat=cost_base * float(rng.lognormvariate(0.0, 0.06)),
            schedule_slots_hat=schedule,
            licensed_hat=licensed,
            insured_hat=insured,
        )
        spec_ai[provider_id] = ai

    return tuple(providers), spec_standard, spec_ai


def _make_jobs_for_cell(
    *,
    rng: random.Random,
    city_id: str,
    week: int,
    category: Category,
    cell_id: str,
    n_jobs: int,
    params: FieldV2Params,
) -> tuple[tuple[TruthJob, ...], dict[str, SpecJob], dict[str, SpecJob]]:
    jobs: list[TruthJob] = []
    spec_standard: dict[str, SpecJob] = {}
    spec_ai: dict[str, SpecJob] = {}

    for i in range(n_jobs):
        job_id = f"{cell_id}_c{i:03d}"
        task = _sample_task(rng, category=category)
        weights = _draw_weights_dirichlet(
            rng, k=len(DIMENSIONS), alpha=params.weight_alpha, category=category
        )
        attributes = _dirichlet(rng, len(DIMENSIONS), alpha=params.dirichlet_alpha)
        budget_true = _draw_budget_for_task(rng=rng, task=task)
        complexity = _draw_complexity_for_task(rng=rng, task=task)
        weirdness = _draw_weirdness_for_task(rng=rng, task=task, complexity=complexity)
        if category == "easy":
            schedule = _draw_schedule(
                rng,
                horizon_days=params.horizon_days,
                slots_per_day=params.slots_per_day,
                mean_slots=int(
                    task.schedule_mean_share * params.horizon_days * params.slots_per_day
                ),
            )
        else:
            schedule = _draw_schedule(
                rng,
                horizon_days=params.horizon_days,
                slots_per_day=params.slots_per_day,
                mean_slots=int(
                    task.schedule_mean_share * params.horizon_days * params.slots_per_day
                ),
            )
        requires_license = rng.random() < task.license_prob
        requires_insurance = rng.random() < task.insurance_prob
        tj = TruthJob(
            job_id=job_id,
            city_id=city_id,
            cell_id=cell_id,
            category=category,
            task_id=task.task_id,
            task_label=task.label,
            weights=weights,
            attributes=attributes,
            budget_true=budget_true,
            schedule_slots=schedule,
            requires_license=requires_license,
            requires_insurance=requires_insurance,
            complexity=complexity,
            weirdness=weirdness,
        )
        jobs.append(tj)

        strategic = rng.random() < params.strategic_share
        shade = params.shade_frac if strategic else 0.0
        budget_reported_std = budget_true * (1.0 - shade)
        budget_reported_std = float(round(budget_reported_std / 25.0) * 25.0)
        weights_hat_std = _weights_standard_from_truth(weights)
        if category == "hard" and rng.random() < params.std_weight_misclass_hard:
            k = len(weights)
            top = sorted(range(k), key=lambda idx: weights[idx], reverse=True)[: min(3, k)]
            non_top = [idx for idx in range(k) if idx not in top]
            if top and non_top:
                drop = rng.choice(top)
                add = rng.choice(non_top)
                new_top = [idx for idx in top if idx != drop] + [add]
                pattern = (0.60, 0.25, 0.15)
                out = [0.0 for _ in weights]
                for rank, idx in enumerate(new_top[: len(pattern)]):
                    out[idx] = pattern[rank]
                total = sum(out)
                if total > 0.0:
                    weights_hat_std = tuple(x / total for x in out)
        spec_standard[job_id] = SpecJob(
            job_id=job_id,
            weights_hat=weights_hat_std,
            budget_reported=budget_reported_std,
            schedule_slots_reported=None,
            requires_license_reported=True if requires_license and rng.random() < 0.6 else None,
            requires_insurance_reported=True if requires_insurance and rng.random() < 0.6 else None,
            complexity_reported=max(0.6, complexity - abs(rng.gauss(0.0, 0.25))),
            weirdness_reported=max(0.0, weirdness - abs(rng.gauss(0.0, 0.20))),
            edited=False,
            strategic_shade=strategic,
        )

        shade_ai = shade + (params.ai_shade_uplift if strategic else 0.0)
        budget_reported_ai = (
            budget_true
            * (1.0 - shade_ai)
            * float(rng.lognormvariate(0.0, params.ai_budget_noise_sd))
        )
        schedule_ai = set(schedule)
        drop_n = int(round(len(schedule_ai) * params.ai_schedule_drop))
        if drop_n > 0:
            for slot in rng.sample(list(schedule_ai), k=min(drop_n, len(schedule_ai))):
                schedule_ai.discard(slot)
        add_n = int(round((params.horizon_days * params.slots_per_day) * params.ai_schedule_add))
        if add_n > 0:
            for slot in rng.sample(
                list(range(params.horizon_days * params.slots_per_day)), k=add_n
            ):
                schedule_ai.add(slot)

        req_license_reported = requires_license and rng.random() > params.ai_requirement_fn
        if not requires_license and rng.random() < params.ai_requirement_fp:
            req_license_reported = True
        req_ins_reported = requires_insurance and rng.random() > params.ai_requirement_fn
        if not requires_insurance and rng.random() < params.ai_requirement_fp:
            req_ins_reported = True

        edited = False
        if (
            abs(budget_reported_ai - budget_true) / max(budget_true, 1e-9) > 0.20
            and rng.random() < params.ai_edit_big_budget
        ):
            budget_reported_ai = budget_true * (1.0 - shade_ai)
            edited = True
        if (
            requires_license
            and not req_license_reported
            and rng.random() < params.ai_edit_requirement
        ):
            req_license_reported = True
            edited = True
        if (
            requires_insurance
            and not req_ins_reported
            and rng.random() < params.ai_edit_requirement
        ):
            req_ins_reported = True
            edited = True
        if (
            len(schedule_ai) < max(1, int(0.5 * len(schedule)))
            and rng.random() < params.ai_edit_schedule
        ):
            schedule_ai = set(schedule)
            edited = True

        spec_ai[job_id] = SpecJob(
            job_id=job_id,
            weights_hat=_weights_ai_from_truth(
                rng=rng, weights=weights, noise_sd=params.ai_weight_noise_sd
            ),
            budget_reported=float(round(budget_reported_ai / 10.0) * 10.0),
            schedule_slots_reported=frozenset(schedule_ai),
            requires_license_reported=req_license_reported,
            requires_insurance_reported=req_ins_reported,
            complexity_reported=max(0.6, complexity + rng.gauss(0.0, 0.10)),
            weirdness_reported=_clip01(weirdness + rng.gauss(0.0, 0.10)),
            edited=edited,
            strategic_shade=strategic,
        )

    return tuple(jobs), spec_standard, spec_ai


@dataclass(frozen=True)
class MeasurementStats:
    d_hat_i: float
    d_hat_j: float
    reciprocity: float
    by_rank: dict[int, float]
    accept_decisions: int


def measure_recommendations(
    *,
    rng: random.Random,
    jobs: tuple[TruthJob, ...],
    providers: tuple[TruthProvider, ...],
    spec_jobs: dict[str, SpecJob],
    spec_providers: dict[str, SpecProvider],
    fit_customer_true: list[list[float]],
    fit_provider_true: list[list[float]],
    params: FieldV2Params,
) -> MeasurementStats:
    total_pairs = 0
    accept_i = 0
    accept_j = 0
    accept_both = 0
    by_rank_counts: dict[int, list[int]] = {r: [0, 0] for r in range(1, params.central_rec_k + 1)}

    for i, job in enumerate(jobs):
        sj = spec_jobs[job.job_id]
        # Score providers by predicted mutual fit under observed weights.
        scores = []
        for j, p in enumerate(providers):
            sp = spec_providers[p.provider_id]
            if sj.requires_license_reported and not sp.licensed_hat:
                continue
            if sj.requires_insurance_reported and not sp.insured_hat:
                continue
            if sj.schedule_slots_reported is not None and not _schedule_overlap(
                sj.schedule_slots_reported, sp.schedule_slots_hat
            ):
                continue
            vc_hat = _dot(sj.weights_hat, p.attributes)
            vp_hat = _dot(sp.weights_hat, job.attributes)
            scores.append((vc_hat + vp_hat, j))
        scores.sort(reverse=True)
        recs = [j for _s, j in scores[: params.central_rec_k]]

        for rank, j in enumerate(recs, start=1):
            total_pairs += 1
            p = providers[j]
            sp = spec_providers[p.provider_id]
            price_quote = _quote_price(
                rng=rng,
                job=job,
                sj=sj,
                provider=p,
                sp=sp,
                inbox_load=0,
                params=params,
            )
            cust_ok = _customer_accepts(
                job=job,
                provider=p,
                fit_true=fit_customer_true[i][j],
                price=price_quote,
                params=params,
            )
            prov_ok = _provider_accepts(
                job=job,
                sj=sj,
                provider=p,
                fit_true=fit_provider_true[j][i],
                price=price_quote,
                params=params,
                info_reveal=0.0,
            )
            accept_i += 1 if cust_ok else 0
            accept_j += 1 if prov_ok else 0
            both = cust_ok and prov_ok
            accept_both += 1 if both else 0
            by_rank_counts[rank][0] += 1 if both else 0
            by_rank_counts[rank][1] += 1

    d_i = accept_i / total_pairs if total_pairs else 0.0
    d_j = accept_j / total_pairs if total_pairs else 0.0
    r = accept_both / total_pairs if total_pairs else 0.0
    by_rank = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in by_rank_counts.items()}
    return MeasurementStats(
        d_hat_i=d_i,
        d_hat_j=d_j,
        reciprocity=r,
        by_rank=by_rank,
        accept_decisions=2 * total_pairs,
    )


def _quote_price(
    *,
    rng: random.Random,
    job: TruthJob,
    sj: SpecJob,
    provider: TruthProvider,
    sp: SpecProvider,
    inbox_load: int,
    params: FieldV2Params,
    cost_signal: float | None = None,
    complexity_signal: float | None = None,
) -> float:
    # Provider sets quote with markup increasing in demand and inferred budget.
    demand_factor = max(0.0, (inbox_load / max(1, params.provider_daily_screen_cap)) - 1.0)
    inferred_budget = sj.budget_reported
    base_cost = sp.cost_base_hat if cost_signal is None else cost_signal
    complexity = sj.complexity_reported if complexity_signal is None else complexity_signal
    base = base_cost * max(0.7, complexity)
    markup = params.base_markup + params.demand_markup * demand_factor
    markup += params.budget_markup * math.tanh((inferred_budget - base) / max(base, 1e-9))
    markup += rng.gauss(0.0, 0.03)
    quote = base * (1.0 + max(-0.05, min(0.7, markup)))
    return max(25.0, quote)


def _customer_accepts(
    *,
    job: TruthJob,
    provider: TruthProvider,
    fit_true: float,
    price: float,
    params: FieldV2Params,
) -> bool:
    if job.requires_license and not provider.licensed:
        return False
    if job.requires_insurance and not provider.insured:
        return False
    if not _schedule_overlap(job.schedule_slots, provider.schedule_slots):
        return False
    if price > job.budget_true:
        return False
    return not fit_true < params.accept_threshold


def _provider_accepts(
    *,
    job: TruthJob,
    sj: SpecJob,
    provider: TruthProvider,
    fit_true: float,
    price: float,
    params: FieldV2Params,
    info_reveal: float,
) -> bool:
    tolerance = _clip01(0.15 + 1.4 * provider.attributes[4])
    observed_weirdness = sj.weirdness_reported + info_reveal * (
        job.weirdness - sj.weirdness_reported
    )
    if observed_weirdness > tolerance + 0.10:
        return False
    if not _schedule_overlap(job.schedule_slots, provider.schedule_slots):
        return False
    cost = provider.cost_base * job.complexity
    profit = price - cost
    if profit < 0:
        return False
    return not fit_true < params.accept_threshold


def _cell_preference_density(
    *,
    job_indices: list[int],
    job_by_idx: list[TruthJob],
    providers: tuple[TruthProvider, ...],
    spec_jobs: dict[str, SpecJob],
    spec_providers: dict[str, SpecProvider],
    fit_customer_true: list[list[float]],
    fit_provider_true: list[list[float]],
    epsilon: float,
) -> tuple[float, float, float]:
    """
    Ground-truth analog of the paper's preference density (d_I, d_J):
    how often profile-inferred values are within epsilon of true values.
    """
    total = 0
    close_i = 0
    close_j = 0
    close_both = 0
    for job_idx in job_indices:
        job = job_by_idx[job_idx]
        sj = spec_jobs[job.job_id]
        for p_idx, provider in enumerate(providers):
            sp = spec_providers[provider.provider_id]
            v_i_true = fit_customer_true[job_idx][p_idx]
            v_i_hat = _dot(sj.weights_hat, provider.attributes)
            v_j_true = fit_provider_true[p_idx][job_idx]
            v_j_hat = _dot(sp.weights_hat, job.attributes)
            ok_i = abs(v_i_true - v_i_hat) <= epsilon
            ok_j = abs(v_j_true - v_j_hat) <= epsilon
            close_i += 1 if ok_i else 0
            close_j += 1 if ok_j else 0
            close_both += 1 if ok_i and ok_j else 0
            total += 1
    if total <= 0:
        return 0.0, 0.0, 0.0
    return close_i / total, close_j / total, close_both / total


@dataclass(frozen=True)
class JobOutcome:
    job_id: str
    cell_id: str
    city_id: str
    week: int
    category: Category
    task_id: str
    task_label: str
    elicitation: Elicitation
    mechanism: Mechanism
    ai: int
    central: int
    matched: int
    day_matched: int
    canceled: int
    messages_sent: int
    responses_received: int
    accept_decisions: int
    matched_provider_id: str
    budget_true: float
    budget_reported: float
    schedule_slots_true_n: int
    schedule_slots_reported_n: int
    requires_license_true: int
    requires_license_reported: int
    requires_insurance_true: int
    requires_insurance_reported: int
    complexity_true: float
    complexity_reported: float
    weirdness_true: float
    weirdness_reported: float
    price_paid: float
    consumer_surplus: float
    provider_profit: float
    total_surplus: float
    net_welfare: float
    edited: int
    strategic_shade: int


@dataclass(frozen=True)
class ProviderOutcome:
    provider_id: str
    city_id: str
    week: int
    cost_base: float
    licensed: int
    insured: int
    schedule_slots_n: int
    matches: int
    revenue: float
    profit: float
    inbox: int
    responded: int


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_ols_table(*, result: OLSResult, names: list[str]) -> list[dict[str, RowValue]]:
    rows: list[dict[str, RowValue]] = []
    for idx, name in enumerate(names):
        rows.append(
            {
                "term": name,
                "coef": round(result.coef[idx], 4),
                "se(cluster)": round(result.se[idx], 4),
                "t": round(result.t[idx], 3),
                "p(normal)": round(result.p[idx], 4),
            }
        )
    rows.append(
        {
            "term": "n_obs / n_clusters",
            "coef": f"{result.n_obs} / {result.n_clusters}",
            "se(cluster)": "",
            "t": "",
            "p(normal)": "",
        }
    )
    return rows


def _central_match(
    *,
    rng: random.Random,
    jobs: list[int],
    job_by_idx: list[TruthJob],
    providers: tuple[TruthProvider, ...],
    spec_jobs: dict[str, SpecJob],
    spec_providers: dict[str, SpecProvider],
    fit_customer_true: list[list[float]],
    fit_provider_true: list[list[float]],
    provider_remaining: list[int],
    params: FieldV2Params,
) -> tuple[list[tuple[int, int, float]], int, int]:
    """
    Returns matches as (job_idx, provider_idx, price), plus (accept_decisions, actions_used).
    actions_used counts provider screening actions consumed by recommendation review.
    """
    rec_k = params.central_rec_k

    # Expand providers by remaining capacity.
    provider_clone_to_provider: list[int] = []
    provider_to_clones: dict[int, list[int]] = {}
    for j, cap in enumerate(provider_remaining):
        for _ in range(max(0, cap)):
            clone_idx = len(provider_clone_to_provider)
            provider_clone_to_provider.append(j)
            provider_to_clones.setdefault(j, []).append(clone_idx)
    n_right = len(provider_clone_to_provider)
    if n_right == 0:
        return [], 0, 0

    accepted_adj: list[list[int]] = [[] for _ in jobs]
    accept_decisions = 0
    actions_used = 0

    for local_i, job_idx in enumerate(jobs):
        job = job_by_idx[job_idx]
        sj = spec_jobs[job.job_id]
        scores: list[tuple[float, int]] = []
        for p_idx, p in enumerate(providers):
            if provider_remaining[p_idx] <= 0:
                continue
            sp = spec_providers[p.provider_id]
            if sj.requires_license_reported and not sp.licensed_hat:
                continue
            if sj.requires_insurance_reported and not sp.insured_hat:
                continue
            if sj.schedule_slots_reported is not None and not _schedule_overlap(
                sj.schedule_slots_reported, sp.schedule_slots_hat
            ):
                continue
            vc_hat = _dot(sj.weights_hat, p.attributes)
            vp_hat = _dot(sp.weights_hat, job.attributes)
            scores.append((vc_hat + vp_hat, p_idx))
        scores.sort(reverse=True)
        recs = scores[: min(rec_k, len(scores))]
        for _score, p_idx in recs:
            accept_decisions += 2
            actions_used += 1  # provider must screen this recommendation

            p = providers[p_idx]
            sp = spec_providers[p.provider_id]
            price = _quote_price(
                rng=rng,
                job=job,
                sj=sj,
                provider=p,
                sp=sp,
                inbox_load=0,
                params=params,
            )
            if not _customer_accepts(
                job=job,
                provider=p,
                fit_true=fit_customer_true[job_idx][p_idx],
                price=price,
                params=params,
            ):
                continue
            if not _provider_accepts(
                job=job,
                sj=sj,
                provider=p,
                fit_true=fit_provider_true[p_idx][job_idx],
                price=price,
                params=params,
                info_reveal=params.ai_intake_reveal
                if sj.schedule_slots_reported is not None
                else 0.0,
            ):
                continue
            accepted_adj[local_i].extend(provider_to_clones.get(p_idx, []))

    # Maximum cardinality matching (jobs list is left side).
    match_r = _hopcroft_karp(accepted_adj, n_left=len(jobs), n_right=n_right)
    matches: list[tuple[int, int, float]] = []
    for clone_idx, local_i in enumerate(match_r):
        if local_i == -1:
            continue
        job_idx = jobs[local_i]
        provider_idx = provider_clone_to_provider[clone_idx]
        job = job_by_idx[job_idx]
        sj = spec_jobs[job.job_id]
        p = providers[provider_idx]
        sp = spec_providers[p.provider_id]
        price = _quote_price(
            rng=rng,
            job=job,
            sj=sj,
            provider=p,
            sp=sp,
            inbox_load=0,
            params=params,
        )
        matches.append((job_idx, provider_idx, price))
        provider_remaining[provider_idx] -= 1

    return matches, accept_decisions, actions_used


def _hopcroft_karp(adj: list[list[int]], n_left: int, n_right: int) -> list[int]:
    """
    Returns match_r[right] = left_index or -1; maximum cardinality matching.
    """
    from collections import deque

    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [-1] * n_left

    def bfs() -> bool:
        q: deque[int] = deque()
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                u2 = match_r[v]
                if u2 != -1 and dist[u2] == -1:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
                if u2 == -1:
                    found = True
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            u2 = match_r[v]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                dfs(u)

    return match_r


def simulate_city_week(
    *,
    rng: random.Random,
    assignment: CityWeekAssignment,
    params: FieldV2Params,
    n_jobs_easy: int,
    n_jobs_hard: int,
    n_providers: int,
) -> tuple[
    list[CellOutcome],
    list[JobOutcome],
    list[ProviderOutcome],
    list[dict[str, RowValue]],
]:
    city_id = assignment.city_id
    week = assignment.week

    providers, spec_p_std, spec_p_ai = _make_provider_pool(
        rng=rng, city_id=city_id, n_providers=n_providers, params=params
    )
    provider_remaining = [params.provider_weekly_capacity for _ in providers]

    # Build two cells that share the same providers (spillovers).
    cell_easy_id = f"{city_id}_w{week:02d}_easy"
    cell_hard_id = f"{city_id}_w{week:02d}_hard"

    jobs_easy, spec_easy_std, spec_easy_ai = _make_jobs_for_cell(
        rng=rng,
        city_id=city_id,
        week=week,
        category="easy",
        cell_id=cell_easy_id,
        n_jobs=n_jobs_easy,
        params=params,
    )
    jobs_hard, spec_hard_std, spec_hard_ai = _make_jobs_for_cell(
        rng=rng,
        city_id=city_id,
        week=week,
        category="hard",
        cell_id=cell_hard_id,
        n_jobs=n_jobs_hard,
        params=params,
    )

    job_by_idx: list[TruthJob] = list(jobs_easy) + list(jobs_hard)
    n_jobs_total = len(job_by_idx)
    n_p = len(providers)

    fit_customer_true: list[list[float]] = []
    for job in job_by_idx:
        row = []
        for p in providers:
            base = _dot(job.weights, p.attributes)
            row.append(_clip01(base + rng.gauss(0.0, params.idiosyncratic_noise_sd)))
        fit_customer_true.append(row)
    fit_provider_true: list[list[float]] = []
    for p in providers:
        row = []
        for job in job_by_idx:
            base = _dot(p.weights, job.attributes)
            row.append(_clip01(base + rng.gauss(0.0, params.idiosyncratic_noise_sd)))
        fit_provider_true.append(row)

    # Cell-level assignment determines which spec mapping and which mechanism is offered.
    def realized_elicitation(cell: tuple[Elicitation, Mechanism]) -> Elicitation:
        elic, _mech = cell
        if elic == "ai":
            return "ai" if rng.random() < params.compliance_ai else "standard"
        return "ai" if rng.random() < params.contamination_ai_in_control else "standard"

    def realized_mechanism(mech_assigned: Mechanism) -> Mechanism:
        if rng.random() < params.compliance_central:
            return mech_assigned
        return "search" if mech_assigned == "central" else "central"

    cell_settings = {
        cell_easy_id: assignment.cell_easy,
        cell_hard_id: assignment.cell_hard,
    }

    realized_spec_jobs: dict[str, SpecJob] = {}

    # Provider-side describability (d_J) improves when the platform deploys the AI elicitation stack
    # in a city-week (spillover across simultaneous cells sharing providers).
    any_ai_assigned = any(elic == "ai" for elic, _mech in cell_settings.values())
    realized_spec_providers: dict[str, SpecProvider] = spec_p_ai if any_ai_assigned else spec_p_std

    # Jobs: per-job realized elicitation (with compliance).
    for job in job_by_idx:
        cell = cell_settings[job.cell_id]
        realized = realized_elicitation(cell)
        if job.category == "easy":
            realized_spec_jobs[job.job_id] = (
                spec_easy_ai[job.job_id] if realized == "ai" else spec_easy_std[job.job_id]
            )
        else:
            realized_spec_jobs[job.job_id] = (
                spec_hard_ai[job.job_id] if realized == "ai" else spec_hard_std[job.job_id]
            )

    realized_cell_mechanism: dict[str, Mechanism] = {
        cell_easy_id: realized_mechanism(cell_settings[cell_easy_id][1]),
        cell_hard_id: realized_mechanism(cell_settings[cell_hard_id][1]),
    }

    # Dynamic state.
    matched_provider: list[int | None] = [None for _ in range(n_jobs_total)]
    day_matched: list[int | None] = [None for _ in range(n_jobs_total)]
    canceled: list[int] = [0 for _ in range(n_jobs_total)]
    price_paid: list[float] = [0.0 for _ in range(n_jobs_total)]
    messages_sent: list[int] = [0 for _ in range(n_jobs_total)]
    responses_received: list[int] = [0 for _ in range(n_jobs_total)]
    accept_decisions: list[int] = [0 for _ in range(n_jobs_total)]
    contacted: list[set[int]] = [set() for _ in range(n_jobs_total)]

    provider_inbox_total = [0 for _ in range(n_p)]
    provider_responded_total = [0 for _ in range(n_p)]
    provider_revenue = [0.0 for _ in range(n_p)]
    provider_profit_total = [0.0 for _ in range(n_p)]
    provider_matches = [0 for _ in range(n_p)]

    # Simulate day-by-day; central cells run matching on specified days.
    for day in range(1, params.horizon_days + 1):
        # Provider screening budget per day combines search inbox + rec review.
        provider_screen_left = [params.provider_daily_screen_cap for _ in range(n_p)]

        # Central matching events.
        for cell_id in (cell_easy_id, cell_hard_id):
            if realized_cell_mechanism[cell_id] != "central" or day not in params.central_days:
                continue
            jobs_in_cell = [
                idx
                for idx, job in enumerate(job_by_idx)
                if job.cell_id == cell_id and matched_provider[idx] is None
            ]
            if not jobs_in_cell:
                continue
            matches, decs, actions_used = _central_match(
                rng=rng,
                jobs=jobs_in_cell,
                job_by_idx=job_by_idx,
                providers=providers,
                spec_jobs=realized_spec_jobs,
                spec_providers=realized_spec_providers,
                fit_customer_true=fit_customer_true,
                fit_provider_true=fit_provider_true,
                provider_remaining=provider_remaining,
                params=params,
            )
            # Provider screening capacity consumed.
            # Spread actions_used across providers heuristically (in practice it is per-rec).
            # Here: subtract 1 screen action from providers involved, else from busiest.
            for _ in range(actions_used):
                j = rng.randrange(n_p)
                if provider_screen_left[j] > 0:
                    provider_screen_left[j] -= 1
            for job_idx, provider_idx, price in matches:
                matched_provider[job_idx] = provider_idx
                day_matched[job_idx] = day
                price_paid[job_idx] = price
                accept_decisions[job_idx] += decs // max(1, len(jobs_in_cell))

        # Search requests.
        inbox: list[list[int]] = [[] for _ in range(n_p)]
        for job_idx, job in enumerate(job_by_idx):
            if matched_provider[job_idx] is not None:
                continue
            cell_id = job.cell_id
            if realized_cell_mechanism[cell_id] != "search":
                continue

            k = params.search_k_per_day
            scored: list[tuple[float, int]] = []
            for p_idx, p in enumerate(providers):
                if provider_remaining[p_idx] <= 0 or p_idx in contacted[job_idx]:
                    continue
                # In decentralized search, the customer can directly evaluate providers using their
                # *true* preferences and publicly observable provider attributes/credentials.
                if job.requires_license and not p.licensed:
                    continue
                if job.requires_insurance and not p.insured:
                    continue
                score = _dot(job.weights, p.attributes)
                score += rng.gauss(0.0, 0.05)
                scored.append((score, p_idx))
            scored.sort(reverse=True)
            k_eff = min(k, len(scored))
            if k_eff <= 0:
                continue
            explore = max(0.0, min(1.0, params.search_explore_share))
            top_n = int(round(k_eff * (1.0 - explore)))
            if top_n <= 0:
                top_n = 1
            chosen = [p_idx for _score, p_idx in scored[:top_n]]
            remaining = [p_idx for _score, p_idx in scored[top_n:]]
            if k_eff > len(chosen) and remaining:
                chosen.extend(rng.sample(remaining, k=min(k_eff - len(chosen), len(remaining))))
            for p_idx in chosen:
                inbox[p_idx].append(job_idx)
                contacted[job_idx].add(p_idx)
                messages_sent[job_idx] += 1

        # Providers respond (subject to daily screening cap).
        responses: list[list[tuple[int, float]]] = [[] for _ in range(n_jobs_total)]
        for p_idx, incoming in enumerate(inbox):
            if not incoming or provider_remaining[p_idx] <= 0:
                continue
            provider_inbox_total[p_idx] += len(incoming)
            provider_screen_left[p_idx] = max(0, provider_screen_left[p_idx])

            p = providers[p_idx]
            candidates: list[tuple[float, int, float]] = []
            for job_idx in incoming:
                if provider_screen_left[p_idx] <= 0:
                    break
                job = job_by_idx[job_idx]
                sj = realized_spec_jobs[job.job_id]

                # In search, early communication reveals hard feasibility quickly.
                if job.requires_license and not p.licensed:
                    continue
                if job.requires_insurance and not p.insured:
                    continue
                if not _schedule_overlap(job.schedule_slots, p.schedule_slots):
                    continue

                quote = _quote_price(
                    rng=rng,
                    job=job,
                    sj=sj,
                    provider=p,
                    sp=realized_spec_providers[p.provider_id],
                    inbox_load=len(incoming),
                    params=params,
                    cost_signal=p.cost_base,
                    complexity_signal=job.complexity,
                )
                expected_cost = p.cost_base * job.complexity
                expected_profit = quote - expected_cost
                if expected_profit < 0:
                    continue
                score = expected_profit + 0.2 * _dot(p.weights, job.attributes)
                score += rng.gauss(0.0, 0.05)
                candidates.append((score, job_idx, quote))

            candidates.sort(reverse=True)
            for _score, job_idx, quote in candidates:
                if provider_screen_left[p_idx] <= 0:
                    break
                provider_screen_left[p_idx] -= 1
                provider_responded_total[p_idx] += 1
                responses[job_idx].append((p_idx, quote))
                responses_received[job_idx] += 1

        # Jobs accept best quote and providers finalize under capacity.
        accepters_by_provider: list[list[tuple[float, int, float]]] = [[] for _ in range(n_p)]
        for job_idx, offers in enumerate(responses):
            if matched_provider[job_idx] is not None or not offers:
                continue
            job = job_by_idx[job_idx]
            # Choose offer maximizing customer surplus among feasible offers.
            best: tuple[int, float, float] | None = None
            for p_idx, quote in offers:
                p = providers[p_idx]
                if not _customer_accepts(
                    job=job,
                    provider=p,
                    fit_true=fit_customer_true[job_idx][p_idx],
                    price=quote,
                    params=params,
                ):
                    continue
                gross = (
                    params.value_scale_easy if job.category == "easy" else params.value_scale_hard
                ) * float(fit_customer_true[job_idx][p_idx])
                surplus = gross - quote
                if best is None or surplus > best[2]:
                    best = (p_idx, quote, surplus)
            if best is None:
                continue
            p_idx, quote, surplus = best
            accepters_by_provider[p_idx].append((surplus, job_idx, quote))

        for p_idx, accepters in enumerate(accepters_by_provider):
            if not accepters or provider_remaining[p_idx] <= 0:
                continue
            accepters.sort(reverse=True)
            p = providers[p_idx]
            for _surplus, job_idx, quote in accepters:
                if provider_remaining[p_idx] <= 0 or matched_provider[job_idx] is not None:
                    continue
                job = job_by_idx[job_idx]
                sj = realized_spec_jobs[job.job_id]
                info_reveal = 0.0
                if realized_cell_mechanism[job.cell_id] == "search":
                    actions = messages_sent[job_idx] + responses_received[job_idx]
                    info_reveal = 1.0 - math.exp(-params.search_reveal_rate * float(actions))
                if sj.schedule_slots_reported is not None:
                    info_reveal = max(info_reveal, params.ai_intake_reveal)
                if not _provider_accepts(
                    job=job,
                    sj=sj,
                    provider=p,
                    fit_true=fit_provider_true[p_idx][job_idx],
                    price=quote,
                    params=params,
                    info_reveal=info_reveal,
                ):
                    continue
                matched_provider[job_idx] = p_idx
                day_matched[job_idx] = day
                price_paid[job_idx] = quote
                provider_remaining[p_idx] -= 1

        # Post-match cancellation / reneging: exogenous shocks plus hidden mismatch that wasn't
        # resolved before agreement (renegotiation after under-stated complexity; "weirdness"
        # revealed after acceptance).
        for job_idx, provider_idx_opt in enumerate(matched_provider):
            if provider_idx_opt is None:
                continue
            if day_matched[job_idx] != day:
                continue
            job = job_by_idx[job_idx]
            sj = realized_spec_jobs[job.job_id]
            provider_idx = provider_idx_opt
            p = providers[provider_idx]

            cancel_p = params.cancel_base

            # Hidden "weirdness" revealed after acceptance; higher communication reveals more.
            actions = messages_sent[job_idx] + responses_received[job_idx]
            reveal = 1.0 - math.exp(-params.search_reveal_rate * float(actions))
            if sj.schedule_slots_reported is not None:
                reveal = max(reveal, params.ai_intake_reveal)
            tolerance = _clip01(0.15 + 1.4 * p.attributes[4])
            if job.weirdness > tolerance + 0.10:
                gap = job.weirdness - (tolerance + 0.10)
                cancel_p += params.cancel_weirdness_reveal * gap * (1.0 - reveal)

            # Renegotiation if complexity underreported -> final price revision.
            final_price = price_paid[job_idx]
            effective_complexity = sj.complexity_reported + reveal * (
                job.complexity - sj.complexity_reported
            )
            if effective_complexity < job.complexity:
                bump = (job.complexity - effective_complexity) * 0.12
                final_price *= 1.0 + max(0.0, bump)
                if final_price > job.budget_true:
                    cancel_p += params.cancel_price_reneg
            price_paid[job_idx] = final_price

            if rng.random() < min(0.95, cancel_p):
                canceled[job_idx] += 1
                matched_provider[job_idx] = None
                day_matched[job_idx] = None
                # Capacity is restored; cancellation occurs early enough.
                provider_remaining[provider_idx] += 1
                price_paid[job_idx] = 0.0

    # Compute outcomes and datasets.
    job_rows: list[JobOutcome] = []
    for job_idx, job in enumerate(job_by_idx):
        provider_idx_opt = matched_provider[job_idx]
        matched = 1 if provider_idx_opt is not None else 0
        day_m = int(day_matched[job_idx] or 0)
        price = price_paid[job_idx] if matched else 0.0
        if matched:
            if provider_idx_opt is None:
                raise RuntimeError("matched invariant violated: provider_idx is None")
            provider_idx = provider_idx_opt
            fit_c = fit_customer_true[job_idx][provider_idx]
            gross = (
                params.value_scale_easy if job.category == "easy" else params.value_scale_hard
            ) * float(fit_c)
            consumer_surplus = gross - price
            cost = providers[provider_idx].cost_base * job.complexity
            profit = price - cost
            total_surplus = gross - cost
        else:
            consumer_surplus = 0.0
            profit = 0.0
            total_surplus = 0.0

        actions = messages_sent[job_idx] + responses_received[job_idx] + accept_decisions[job_idx]
        net_welfare = total_surplus - params.attention_cost * actions

        sj = realized_spec_jobs[job.job_id]
        cell_setting = cell_settings[job.cell_id]
        ai_assigned = 1 if cell_setting[0] == "ai" else 0
        central_assigned = 1 if cell_setting[1] == "central" else 0

        job_rows.append(
            JobOutcome(
                job_id=job.job_id,
                cell_id=job.cell_id,
                city_id=job.city_id,
                week=week,
                category=job.category,
                task_id=job.task_id,
                task_label=job.task_label,
                elicitation=cell_setting[0],
                mechanism=cell_setting[1],
                ai=ai_assigned,
                central=central_assigned,
                matched=matched,
                day_matched=day_m,
                canceled=canceled[job_idx],
                messages_sent=messages_sent[job_idx],
                responses_received=responses_received[job_idx],
                accept_decisions=accept_decisions[job_idx],
                matched_provider_id=(
                    providers[provider_idx].provider_id if provider_idx_opt is not None else ""
                ),
                budget_true=round(job.budget_true, 2),
                budget_reported=round(sj.budget_reported, 2),
                schedule_slots_true_n=len(job.schedule_slots),
                schedule_slots_reported_n=(
                    len(sj.schedule_slots_reported)
                    if sj.schedule_slots_reported is not None
                    else -1
                ),
                requires_license_true=1 if job.requires_license else 0,
                requires_license_reported=(
                    -1
                    if sj.requires_license_reported is None
                    else 1
                    if sj.requires_license_reported
                    else 0
                ),
                requires_insurance_true=1 if job.requires_insurance else 0,
                requires_insurance_reported=(
                    -1
                    if sj.requires_insurance_reported is None
                    else 1
                    if sj.requires_insurance_reported
                    else 0
                ),
                complexity_true=round(job.complexity, 3),
                complexity_reported=round(sj.complexity_reported, 3),
                weirdness_true=round(job.weirdness, 3),
                weirdness_reported=round(sj.weirdness_reported, 3),
                price_paid=round(price, 3),
                consumer_surplus=round(consumer_surplus, 3),
                provider_profit=round(profit, 3),
                total_surplus=round(total_surplus, 3),
                net_welfare=round(net_welfare, 3),
                edited=1 if sj.edited else 0,
                strategic_shade=1 if sj.strategic_shade else 0,
            )
        )

        if provider_idx_opt is not None:
            provider_matches[provider_idx_opt] += 1
            provider_revenue[provider_idx_opt] += price
            provider_profit_total[provider_idx_opt] += profit

    provider_rows: list[ProviderOutcome] = []
    for p_idx, p in enumerate(providers):
        provider_rows.append(
            ProviderOutcome(
                provider_id=p.provider_id,
                city_id=p.city_id,
                week=week,
                cost_base=round(p.cost_base, 2),
                licensed=1 if p.licensed else 0,
                insured=1 if p.insured else 0,
                schedule_slots_n=len(p.schedule_slots),
                matches=provider_matches[p_idx],
                revenue=round(provider_revenue[p_idx], 3),
                profit=round(provider_profit_total[p_idx], 3),
                inbox=provider_inbox_total[p_idx],
                responded=provider_responded_total[p_idx],
            )
        )

    # Measurement proxy (instrumented accept/reject) per cell.
    measurement_rows: list[dict[str, RowValue]] = []
    outcomes: list[CellOutcome] = []

    for cell_id, (elic_assigned, mech_assigned) in cell_settings.items():
        jobs_cell = [j for j in job_by_idx if j.cell_id == cell_id]
        job_indices = [idx for idx, j in enumerate(job_by_idx) if j.cell_id == cell_id]

        ms = measure_recommendations(
            rng=random.Random(rng.randrange(1_000_000_000)),  # nosec B311
            jobs=tuple(jobs_cell),
            providers=providers,
            spec_jobs=realized_spec_jobs,
            spec_providers=realized_spec_providers,
            fit_customer_true=[fit_customer_true[idx] for idx in job_indices],
            fit_provider_true=[
                [fit_provider_true[p_idx][idx] for idx in job_indices] for p_idx in range(n_p)
            ],
            params=params,
        )

        for rank, rr in ms.by_rank.items():
            measurement_rows.append(
                {
                    "cell_id": cell_id,
                    "city_id": city_id,
                    "week": week,
                    "category": "easy" if "easy" in cell_id else "hard",
                    "elicitation": elic_assigned,
                    "mechanism": mech_assigned,
                    "ai": 1 if elic_assigned == "ai" else 0,
                    "central": 1 if mech_assigned == "central" else 0,
                    "rank": rank,
                    "reciprocity_rate": round(rr, 6),
                }
            )

        job_outcomes_cell = [r for r in job_rows if r.cell_id == cell_id]
        provider_outcomes_city = provider_rows
        prices = [float(r.price_paid) for r in job_outcomes_cell if r.matched == 1]
        cancel_events = sum(1 for r in job_outcomes_cell if r.canceled > 0)
        match_events = sum(r.matched for r in job_outcomes_cell)
        mean_days = (
            _mean([float(r.day_matched) for r in job_outcomes_cell if r.matched == 1])
            if match_events
            else float(params.horizon_days)
        )
        messages = sum(r.messages_sent for r in job_outcomes_cell)
        inbox_total = sum(p.inbox for p in provider_outcomes_city)
        responded_total = sum(p.responded for p in provider_outcomes_city)
        provider_response_rate = responded_total / inbox_total if inbox_total else 0.0
        provider_inbox_per_day = inbox_total / max(1.0, float(n_p * params.horizon_days))

        consumer_surplus = sum(float(r.consumer_surplus) for r in job_outcomes_cell) / max(
            1, len(job_outcomes_cell)
        )
        provider_profit_cell = sum(float(r.provider_profit) for r in job_outcomes_cell) / max(
            1, len(job_outcomes_cell)
        )
        total_surplus = sum(float(r.total_surplus) for r in job_outcomes_cell) / max(
            1, len(job_outcomes_cell)
        )
        net_welfare = sum(float(r.net_welfare) for r in job_outcomes_cell) / max(
            1, len(job_outcomes_cell)
        )

        herf = _herfindahl([p.matches for p in provider_outcomes_city])

        dens_i, dens_j, dens_both = _cell_preference_density(
            job_indices=job_indices,
            job_by_idx=job_by_idx,
            providers=providers,
            spec_jobs=realized_spec_jobs,
            spec_providers=realized_spec_providers,
            fit_customer_true=fit_customer_true,
            fit_provider_true=fit_provider_true,
            epsilon=params.epsilon,
        )

        outcomes.append(
            CellOutcome(
                cell_id=cell_id,
                city_id=city_id,
                week=week,
                category="easy" if "easy" in cell_id else "hard",
                elicitation=elic_assigned,
                mechanism=mech_assigned,
                n_jobs=len(job_outcomes_cell),
                match_rate=match_events / max(1, len(job_outcomes_cell)),
                mean_days_to_match=mean_days,
                cancel_rate=cancel_events / max(1, match_events) if match_events else 0.0,
                messages_per_job=messages / max(1, len(job_outcomes_cell)),
                provider_inbox_per_day=provider_inbox_per_day,
                provider_response_rate=provider_response_rate,
                avg_price=_mean(prices) if prices else 0.0,
                price_sd=_se(prices) * math.sqrt(len(prices)) if len(prices) >= 2 else 0.0,
                consumer_surplus_per_job=consumer_surplus,
                provider_profit_per_job=provider_profit_cell,
                total_surplus_per_job=total_surplus,
                net_welfare_per_job=net_welfare,
                herfindahl_providers=herf,
                reciprocity_obs=ms.reciprocity,
                pref_density_i=dens_i,
                pref_density_j=dens_j,
                pref_density_both=dens_both,
            )
        )

    return outcomes, job_rows, provider_rows, measurement_rows


def _cells_summary_table(
    *, cells: list[CellOutcome], category: Category
) -> list[dict[str, RowValue]]:
    arms: list[tuple[Elicitation, Mechanism]] = [
        ("standard", "search"),
        ("standard", "central"),
        ("ai", "search"),
        ("ai", "central"),
    ]
    rows: list[dict[str, RowValue]] = []
    for elicitation, mechanism in arms:
        arm = _arm_label(elicitation, mechanism)
        subset = [
            c
            for c in cells
            if c.category == category and _arm_label(c.elicitation, c.mechanism) == arm
        ]
        rows.append(
            {
                "category": category,
                "arm": arm,
                "match_rate": round(_mean([c.match_rate for c in subset]), 3),
                "match_rate_se": round(_se([c.match_rate for c in subset]), 3),
                "cancel_rate": round(_mean([c.cancel_rate for c in subset]), 3),
                "pref_density_i": round(_mean([c.pref_density_i for c in subset]), 3),
                "pref_density_j": round(_mean([c.pref_density_j for c in subset]), 3),
                "pref_density_both": round(_mean([c.pref_density_both for c in subset]), 3),
                "messages_per_job": round(_mean([c.messages_per_job for c in subset]), 3),
                "provider_inbox_per_day": round(
                    _mean([c.provider_inbox_per_day for c in subset]), 3
                ),
                "avg_price": round(_mean([c.avg_price for c in subset]), 2),
                "consumer_surplus": round(_mean([c.consumer_surplus_per_job for c in subset]), 2),
                "provider_profit": round(_mean([c.provider_profit_per_job for c in subset]), 2),
                "net_welfare": round(_mean([c.net_welfare_per_job for c in subset]), 2),
                "reciprocity": round(_mean([c.reciprocity_obs for c in subset]), 3),
                "herfindahl": round(_mean([c.herfindahl_providers for c in subset]), 3),
            }
        )
    return rows


def _run_job_regressions(
    *, jobs: list[JobOutcome], out_dir: Path, suffix: str
) -> dict[str, tuple[list[dict[str, RowValue]], OLSResult]]:
    rows_by_name: dict[str, tuple[list[dict[str, RowValue]], OLSResult]] = {}

    def run(outcome: str) -> tuple[list[dict[str, RowValue]], OLSResult]:
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
        names = [
            "intercept",
            "ai",
            "central",
            "ai_x_central",
            "hard",
            "ai_x_hard",
            "central_x_hard",
            "ai_x_central_x_hard",
        ]
        return _format_ols_table(result=res, names=names), res

    for outcome in ("matched", "net_welfare", "canceled", "consumer_surplus", "provider_profit"):
        table, res = run(outcome)
        _write_md_table(table, out_dir / f"reg_{outcome}{suffix}.md")
        _write_csv(table, out_dir / f"reg_{outcome}{suffix}.csv")
        rows_by_name[outcome] = (table, res)

    return rows_by_name


def _spillover_table(*, cells: list[CellOutcome]) -> list[dict[str, RowValue]]:
    """
    A simple diagnostic: compare outcomes for a cell when the *other category* in the same
    (city, week) is assigned to central/ai vs not.
    """
    by_city_week: dict[tuple[str, int], list[CellOutcome]] = {}
    for c in cells:
        by_city_week.setdefault((c.city_id, c.week), []).append(c)

    rows: list[dict[str, RowValue]] = []
    for category in ("easy", "hard"):
        control_cells = [
            cw
            for cw in by_city_week.values()
            if any(c.category == category for c in cw) and len(cw) == 2
        ]
        treated_neighbor = []
        control_neighbor = []
        for cw in control_cells:
            own = next(c for c in cw if c.category == category)
            other = next(c for c in cw if c.category != category)
            neighbor_treated = 1 if other.elicitation == "ai" or other.mechanism == "central" else 0
            if neighbor_treated:
                treated_neighbor.append(own)
            else:
                control_neighbor.append(own)

        rows.append(
            {
                "category": category,
                "n_city_weeks": len(control_cells),
                "match_rate_neighbor_treated": round(
                    _mean([c.match_rate for c in treated_neighbor]), 3
                ),
                "match_rate_neighbor_control": round(
                    _mean([c.match_rate for c in control_neighbor]), 3
                ),
                "inbox_neighbor_treated": round(
                    _mean([c.provider_inbox_per_day for c in treated_neighbor]), 3
                ),
                "inbox_neighbor_control": round(
                    _mean([c.provider_inbox_per_day for c in control_neighbor]), 3
                ),
            }
        )
    return rows


def run_scaling_sweep(
    *,
    seed: int,
    out_dir: Path,
    params: FieldV2Params,
    sizes: list[int],
    n_cities: int,
    n_weeks: int,
) -> None:
    rng = random.Random(seed)  # nosec B311
    rows: list[dict[str, RowValue]] = []

    for n_jobs in sizes:
        n_providers = max(20, n_jobs)
        sim_cells: list[CellOutcome] = []
        for city_idx in range(n_cities):
            city_id = f"scale_city{city_idx:03d}"
            for week in range(n_weeks):
                assignment = CityWeekAssignment(
                    city_id=city_id,
                    week=week,
                    cell_easy=("standard", "search"),
                    cell_hard=("ai", "central"),
                )
                city_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
                cells, _jobs, _prov, _meas = simulate_city_week(
                    rng=city_rng,
                    assignment=assignment,
                    params=params,
                    n_jobs_easy=n_jobs,
                    n_jobs_hard=n_jobs,
                    n_providers=n_providers,
                )
                sim_cells.extend(cells)

        for category in ("easy", "hard"):
            subset = [c for c in sim_cells if c.category == category]
            rows.append(
                {
                    "n_jobs_per_cell": n_jobs,
                    "n_providers_city": n_providers,
                    "category": category,
                    "match_rate": round(_mean([c.match_rate for c in subset]), 3),
                    "messages_per_job": round(_mean([c.messages_per_job for c in subset]), 3),
                    "provider_inbox_per_day": round(
                        _mean([c.provider_inbox_per_day for c in subset]), 3
                    ),
                    "net_welfare": round(_mean([c.net_welfare_per_job for c in subset]), 2),
                }
            )

    _write_csv(rows, out_dir / "scaling_summary.csv")

    for metric in ("messages_per_job", "provider_inbox_per_day", "net_welfare", "match_rate"):
        series = []
        for category in ("easy", "hard"):
            pts = []
            for n_jobs in sizes:
                vals = [
                    _as_float(r[metric])
                    for r in rows
                    if r["category"] == category and int(_as_float(r["n_jobs_per_cell"])) == n_jobs
                ]
                pts.append((float(n_jobs), _mean(vals)))
            series.append(LineSeries(label=category, points=pts))

        write_line_chart_svg(
            out_path=out_dir / f"fig_scaling_{metric}.svg",
            title=f"Scaling: {metric}",
            x_label="jobs per cell (market size)",
            y_label=metric,
            series=series,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/field_v2_latest")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--cities", type=int, default=120)
    parser.add_argument("--weeks", type=int, default=2)
    parser.add_argument("--jobs-easy", type=int, default=60)
    parser.add_argument("--jobs-hard", type=int, default=60)
    parser.add_argument("--providers", type=int, default=70)
    parser.add_argument("--attention-cost", type=float, default=0.25)
    parser.add_argument("--params-json", default="")
    parser.add_argument("--skip-scaling", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_params = FieldV2Params(attention_cost=args.attention_cost)
    if args.params_json:
        overrides = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        params = _params_with_overrides(base_params, overrides)
    else:
        params = base_params
    rng = random.Random(args.seed)  # nosec B311

    # Balanced assignment within category across all city-weeks.
    arms: list[tuple[Elicitation, Mechanism]] = [
        ("standard", "search"),
        ("standard", "central"),
        ("ai", "search"),
        ("ai", "central"),
    ]

    cell_keys: list[tuple[str, int]] = [
        (f"city{city_idx:03d}", week)
        for city_idx in range(args.cities)
        for week in range(args.weeks)
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

    cells: list[CellOutcome] = []
    jobs: list[JobOutcome] = []
    providers: list[ProviderOutcome] = []
    measurement_rows: list[dict[str, RowValue]] = []

    for a in assignments:
        city_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
        cell_out, job_out, provider_out, meas = simulate_city_week(
            rng=city_rng,
            assignment=a,
            params=params,
            n_jobs_easy=args.jobs_easy,
            n_jobs_hard=args.jobs_hard,
            n_providers=args.providers,
        )
        cells.extend(cell_out)
        jobs.extend(job_out)
        providers.extend(provider_out)
        measurement_rows.extend(meas)

    cell_rows = [asdict(c) for c in cells]
    job_rows = [asdict(j) for j in jobs]
    provider_rows = [asdict(p) for p in providers]

    _write_csv([{k: v for k, v in r.items()} for r in cell_rows], out_dir / "cells.csv")
    _write_csv([{k: v for k, v in r.items()} for r in job_rows], out_dir / "jobs.csv")
    _write_csv([{k: v for k, v in r.items()} for r in provider_rows], out_dir / "providers.csv")
    _write_csv(measurement_rows, out_dir / "reciprocity_by_rank.csv")

    summary_rows = _cells_summary_table(cells=cells, category="easy") + _cells_summary_table(
        cells=cells, category="hard"
    )
    _write_md_table(summary_rows, out_dir / "arm_summary.md")
    _write_csv(summary_rows, out_dir / "arm_summary.csv")

    reg = _run_job_regressions(jobs=jobs, out_dir=out_dir, suffix="")

    spill_rows = _spillover_table(cells=cells)
    _write_md_table(spill_rows, out_dir / "spillovers.md")
    _write_csv(spill_rows, out_dir / "spillovers.csv")

    for category in ("easy", "hard"):
        bars_match = []
        bars_welfare = []
        bars_cancel = []
        bars_density = []
        for elicitation, mechanism in arms:
            arm = _arm_label(elicitation, mechanism)
            subset = [
                c
                for c in cells
                if c.category == category and _arm_label(c.elicitation, c.mechanism) == arm
            ]
            bars_match.append(Bar(label=arm, value=_mean([c.match_rate for c in subset])))
            bars_welfare.append(
                Bar(label=arm, value=_mean([c.net_welfare_per_job for c in subset]))
            )
            bars_cancel.append(Bar(label=arm, value=_mean([c.cancel_rate for c in subset])))
            bars_density.append(Bar(label=arm, value=_mean([c.pref_density_both for c in subset])))

        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_match_rate.svg",
            title=f"Match rate by arm (category={category})",
            bars=bars_match,
            y_label="match_rate",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_net_welfare.svg",
            title=f"Net welfare per job by arm (category={category})",
            bars=bars_welfare,
            y_label="net_welfare_per_job",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_cancel_rate.svg",
            title=f"Cancellation rate by arm (category={category})",
            bars=bars_cancel,
            y_label="cancel_rate",
        )
        write_bar_chart_svg(
            out_path=out_dir / f"fig_{category}_pref_density.svg",
            title=f"Preference density proxy by arm (category={category})",
            bars=bars_density,
            y_label="P(|v_hat - v_true| <= epsilon) on both sides",
        )

        def series_for(arm: tuple[Elicitation, Mechanism], *, cat: Category) -> LineSeries:
            elicitation, mechanism = arm
            pts = []
            for rank in range(1, params.central_rec_k + 1):
                vals = [
                    _as_float(r["reciprocity_rate"])
                    for r in measurement_rows
                    if r["category"] == cat
                    and r["elicitation"] == elicitation
                    and r["mechanism"] == mechanism
                    and int(_as_float(r["rank"])) == rank
                ]
                pts.append((float(rank), _mean(vals)))
            return LineSeries(label=_arm_label(elicitation, mechanism), points=pts)

        write_line_chart_svg(
            out_path=out_dir / f"fig_{category}_reciprocity_curve.svg",
            title=f"Reciprocity proxy vs rec rank (category={category})",
            x_label="rank (1=best)",
            y_label="P(both accept)",
            series=[series_for(a, cat=category) for a in arms],
        )

    # Quick read numbers (DiD and triple interaction).
    def arm_stat(category: Category, arm: str, key: str) -> float:
        for row in summary_rows:
            if row["category"] == category and row["arm"] == arm:
                return _as_float(row[key])
        raise KeyError(f"Missing {category} {arm} {key}")

    did_hard_match = (
        arm_stat("hard", "ai_central", "match_rate")
        - arm_stat("hard", "standard_central", "match_rate")
        - (
            arm_stat("hard", "ai_search", "match_rate")
            - arm_stat("hard", "standard_search", "match_rate")
        )
    )
    did_easy_match = (
        arm_stat("easy", "ai_central", "match_rate")
        - arm_stat("easy", "standard_central", "match_rate")
        - (
            arm_stat("easy", "ai_search", "match_rate")
            - arm_stat("easy", "standard_search", "match_rate")
        )
    )
    did_hard_welfare = (
        arm_stat("hard", "ai_central", "net_welfare")
        - arm_stat("hard", "standard_central", "net_welfare")
        - (
            arm_stat("hard", "ai_search", "net_welfare")
            - arm_stat("hard", "standard_search", "net_welfare")
        )
    )
    did_easy_welfare = (
        arm_stat("easy", "ai_central", "net_welfare")
        - arm_stat("easy", "standard_central", "net_welfare")
        - (
            arm_stat("easy", "ai_search", "net_welfare")
            - arm_stat("easy", "standard_search", "net_welfare")
        )
    )

    triple_idx = 7
    triple = reg["matched"][1].coef[triple_idx]
    triple_se = reg["matched"][1].se[triple_idx]
    triple_p = reg["matched"][1].p[triple_idx]
    triple_w = reg["net_welfare"][1].coef[triple_idx]
    triple_w_se = reg["net_welfare"][1].se[triple_idx]
    triple_w_p = reg["net_welfare"][1].p[triple_idx]

    readme = "\n".join(
        [
            "# FieldSim v2 (dynamic + pricing + cancellations + spillovers + scaling)",
            "",
            "So what: this is a more realistic simulation of the field experiment design, with",
            "endogenous communication/congestion, pricing/negotiation, cancellations/rematching,",
            "strategic budget shading, imperfect AI specs with user edits, and provider spillovers",
            "across simultaneous cells sharing the same providers.",
            "",
            f"Quick read (attention_cost={params.attention_cost}):",
            f"- match_rate DiD (easy): {did_easy_match:+.3f}",
            f"- match_rate DiD (hard): {did_hard_match:+.3f}",
            f"- triple interaction (ai×central×hard): {triple:+.3f}",
            f"  (SE={triple_se:.3f}, p={triple_p:.3f})",
            f"- net_welfare DiD (easy): {did_easy_welfare:+.2f}",
            f"- net_welfare DiD (hard): {did_hard_welfare:+.2f}",
            f"- triple interaction on net_welfare: {triple_w:+.2f}",
            f"  (SE={triple_w_se:.2f}, p={triple_w_p:.3f})",
            "",
            "Key realism features:",
            "- Pricing: provider quotes depend on cost, demand, and inferred budget",
            "- Constraints: license/insurance + schedule overlap (hard feasibility)",
            "- Renegotiation/cancels: under-stated complexity + weirdness revealed post-accept",
            "- Misrep: some customers shade budgets (AI can slightly increase shading)",
            "- Imperfect AI: requirement FP/FN + noisy budget/schedule; edits fix some errors",
            "- Spillovers: easy and hard cells share providers within (city,week)",
            "",
            "Artifacts:",
            "- `cells.csv`, `jobs.csv`, `providers.csv`: synthetic datasets",
            "- `arm_summary.md`: arm means by category",
            "- `reg_*.md`: cluster-robust regressions (cluster=cell)",
            "- `spillovers.md`: neighbor-cell spillover diagnostic",
            "- `fig_*`: main plots",
            "",
            "Run:",
            "- `uv run python -m econ_llm_preferences_experiment.field_sim_v2`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")

    meta = {
        "seed": args.seed,
        "cities": args.cities,
        "weeks": args.weeks,
        "jobs_easy": args.jobs_easy,
        "jobs_hard": args.jobs_hard,
        "providers": args.providers,
        "params": asdict(params),
        "market_params": asdict(MarketParams()),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.skip_scaling:
        run_scaling_sweep(
            seed=args.seed + 1,
            out_dir=out_dir,
            params=params,
            sizes=[20, 40, 80, 160],
            n_cities=25,
            n_weeks=1,
        )

    log(logger, 20, "field_v2_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
