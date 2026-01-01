from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from econ_llm_preferences_experiment.models import MatchOutcome


def _argsort_desc(values: tuple[float, ...]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


@dataclass(frozen=True)
class SearchParams:
    max_rounds: int = 15


def decentralized_search(
    *,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    v_customer_hat: tuple[tuple[float, ...], ...],
    accept_threshold: float,
    params: SearchParams,
) -> MatchOutcome:
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)
    customer_order = [_argsort_desc(v_customer_hat[i]) for i in range(n_c)]
    next_choice = [0 for _ in range(n_c)]
    held_by_provider: list[int | None] = [None for _ in range(n_p)]
    proposals = 0
    rounds = 0

    unmatched_customers = set(range(n_c))

    for r in range(1, params.max_rounds + 1):
        rounds = r
        proposals_this_round: list[list[int]] = [[] for _ in range(n_p)]
        for i in list(unmatched_customers):
            if next_choice[i] >= n_p:
                unmatched_customers.discard(i)
                continue
            j = customer_order[i][next_choice[i]]
            next_choice[i] += 1
            proposals_this_round[j].append(i)
            proposals += 1

        any_activity = any(proposals_this_round)
        if not any_activity:
            break

        for j in range(n_p):
            candidates = proposals_this_round[j]
            held = held_by_provider[j]
            if held is not None:
                candidates.append(held)
            best_i: int | None = None
            best_val = -1.0
            for i in candidates:
                if v_customer_true[i][j] < accept_threshold:
                    continue
                if v_provider_true[j][i] < accept_threshold:
                    continue
                val = v_provider_true[j][i]
                if val > best_val:
                    best_val = val
                    best_i = i
            held_by_provider[j] = best_i

        unmatched_customers = set(range(n_c))
        for held_customer in held_by_provider:
            if held_customer is not None:
                unmatched_customers.discard(held_customer)

    matches = tuple((i, j) for j, i in enumerate(held_by_provider) if i is not None)
    return MatchOutcome(matches=matches, proposals=proposals, accept_decisions=0, rounds=rounds)


def _hopcroft_karp(adj: list[list[int]], n_left: int, n_right: int) -> list[int]:
    """
    Returns match_r[right] = left_index or -1; maximum cardinality matching.
    """
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


@dataclass(frozen=True)
class CentralizedParams:
    rec_k: int = 5


def centralized_recommendations(
    *,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    v_customer_hat: tuple[tuple[float, ...], ...],
    v_provider_hat: tuple[tuple[float, ...], ...],
    accept_threshold: float,
    params: CentralizedParams,
) -> MatchOutcome:
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)

    accept_decisions = 0
    accepted_adj: list[list[int]] = [[] for _ in range(n_c)]

    for i in range(n_c):
        scores = [(v_customer_hat[i][j] + v_provider_hat[j][i], j) for j in range(n_p)]
        scores.sort(reverse=True)
        recs = [j for _, j in scores[: params.rec_k]]
        for j in recs:
            accept_decisions += 2
            if v_customer_true[i][j] < accept_threshold:
                continue
            if v_provider_true[j][i] < accept_threshold:
                continue
            accepted_adj[i].append(j)

    match_r = _hopcroft_karp(accepted_adj, n_left=n_c, n_right=n_p)
    matches = []
    for j, i in enumerate(match_r):
        if i != -1:
            matches.append((i, j))
    return MatchOutcome(
        matches=tuple(matches), proposals=0, accept_decisions=accept_decisions, rounds=1
    )


ProposerSide = Literal["customer", "provider"]


@dataclass(frozen=True)
class DeferredAcceptanceParams:
    """
    Gale–Shapley deferred acceptance (one-to-one) with a designated proposing side.

    Notes:
    - Preferences are induced from the *_hat score matrices (higher score = more preferred).
    - Each side can mark partners unacceptable using its own *_true values and accept_threshold,
      so agents can remain unmatched instead of accepting below-threshold partners.
    - Ties in scores are broken deterministically by lower index.
    """

    proposer_side: ProposerSide = "customer"
    max_rounds: int | None = None
    proposer_list_k: int | None = None
    receiver_list_k: int | None = None


def deferred_acceptance(
    *,
    v_customer_true: tuple[tuple[float, ...], ...],
    v_provider_true: tuple[tuple[float, ...], ...],
    v_customer_hat: tuple[tuple[float, ...], ...],
    v_provider_hat: tuple[tuple[float, ...], ...],
    accept_threshold: float,
    params: DeferredAcceptanceParams,
) -> MatchOutcome:
    proposer_side = params.proposer_side
    n_c = len(v_customer_true)
    n_p = len(v_provider_true)

    if proposer_side == "customer":
        n_prop = n_c
        n_recv = n_p
    else:
        n_prop = n_p
        n_recv = n_c

    proposer_list_k = params.proposer_list_k
    receiver_list_k = params.receiver_list_k

    if proposer_list_k is not None:
        proposer_list_k = max(0, int(proposer_list_k))
    if receiver_list_k is not None:
        receiver_list_k = max(0, int(receiver_list_k))

    def proposer_accepts(prop: int, recv: int) -> bool:
        if proposer_side == "customer":
            return v_customer_true[prop][recv] >= accept_threshold
        return v_provider_true[prop][recv] >= accept_threshold

    def receiver_accepts(recv: int, prop: int) -> bool:
        if proposer_side == "customer":
            return v_provider_true[recv][prop] >= accept_threshold
        return v_customer_true[recv][prop] >= accept_threshold

    def proposer_score(prop: int, recv: int) -> float:
        return (
            v_customer_hat[prop][recv]
            if proposer_side == "customer"
            else v_provider_hat[prop][recv]
        )

    def receiver_score(recv: int, prop: int) -> float:
        return (
            v_provider_hat[recv][prop]
            if proposer_side == "customer"
            else v_customer_hat[recv][prop]
        )

    # Build strict preference lists for proposers (restricted to proposer's acceptability).
    pref_lists: list[list[int]] = []
    for prop in range(n_prop):
        options = [recv for recv in range(n_recv) if proposer_accepts(prop, recv)]
        options.sort(key=lambda recv: (-proposer_score(prop, recv), recv))
        if proposer_list_k is not None:
            options = options[:proposer_list_k]
        pref_lists.append(options)

    # Precompute receiver ranks over acceptable proposers for O(1) comparisons.
    receiver_accept_set: list[set[int]] = []
    receiver_rank: list[list[int]] = []
    for recv in range(n_recv):
        options = [prop for prop in range(n_prop) if receiver_accepts(recv, prop)]
        options.sort(key=lambda prop: (-receiver_score(recv, prop), prop))
        if receiver_list_k is not None:
            options = options[:receiver_list_k]

        receiver_accept_set.append(set(options))
        ranks = [n_prop + 1 for _ in range(n_prop)]
        for r, prop in enumerate(options):
            ranks[prop] = r
        receiver_rank.append(ranks)

    matched_prop_for_recv: list[int | None] = [None for _ in range(n_recv)]
    next_choice = [0 for _ in range(n_prop)]
    unmatched = set(range(n_prop))

    proposals = 0
    rounds = 0

    # Interpret "rounds" as synchronous proposal waves: each currently-unmatched proposer makes
    # exactly one proposal per round (if they have remaining acceptable partners).
    max_rounds = params.max_rounds
    while unmatched:
        if max_rounds is not None and rounds >= max_rounds:
            break
        rounds += 1
        any_proposals = False

        for prop in sorted(unmatched):
            prefs = pref_lists[prop]
            if next_choice[prop] >= len(prefs):
                # Exhausted options; permanently unmatched.
                unmatched.discard(prop)
                continue

            recv = prefs[next_choice[prop]]
            next_choice[prop] += 1
            proposals += 1
            any_proposals = True

            if prop not in receiver_accept_set[recv]:
                continue

            held = matched_prop_for_recv[recv]
            if held is None:
                matched_prop_for_recv[recv] = prop
                unmatched.discard(prop)
                continue

            # Receiver keeps the better proposer under its (hat-based) ranking.
            if receiver_rank[recv][prop] < receiver_rank[recv][held]:
                matched_prop_for_recv[recv] = prop
                unmatched.discard(prop)
                unmatched.add(held)

        if not any_proposals:
            break

    matches: list[tuple[int, int]] = []
    for recv, matched_prop in enumerate(matched_prop_for_recv):
        if matched_prop is None:
            continue
        if proposer_side == "customer":
            matches.append((matched_prop, recv))
        else:
            matches.append((recv, matched_prop))

    return MatchOutcome(
        matches=tuple(matches), proposals=proposals, accept_decisions=0, rounds=rounds
    )
