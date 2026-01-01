from __future__ import annotations

from econ_llm_preferences_experiment.mechanisms import (
    CentralizedParams,
    DeferredAcceptanceParams,
    SearchParams,
    centralized_recommendations,
    decentralized_search,
    deferred_acceptance,
)


def test_matching_outputs_are_feasible() -> None:
    v_c_true = (
        (0.9, 0.1),
        (0.2, 0.8),
    )
    v_p_true = (
        (0.9, 0.2),
        (0.1, 0.8),
    )
    v_c_hat = v_c_true
    v_p_hat = v_p_true

    out_s = decentralized_search(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        accept_threshold=0.5,
        params=SearchParams(max_rounds=3),
    )
    assert len(out_s.matches) <= 2

    out_c = centralized_recommendations(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.5,
        params=CentralizedParams(rec_k=2),
    )
    assert len(out_c.matches) <= 2


def test_deferred_acceptance_customer_proposing_stable() -> None:
    # 2x2 with a classic DA outcome:
    # - Both customers rank provider 0 first.
    # - Provider 0 prefers customer 1; provider 1 prefers customer 0.
    v_c_true = (
        (0.9, 0.9),
        (0.9, 0.9),
    )
    v_p_true = (
        (0.9, 0.9),
        (0.9, 0.9),
    )
    v_c_hat = (
        (0.9, 0.1),
        (0.9, 0.1),
    )
    v_p_hat = (
        (0.1, 0.9),
        (0.9, 0.1),
    )

    out = deferred_acceptance(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.5,
        params=DeferredAcceptanceParams(proposer_side="customer"),
    )
    assert set(out.matches) == {(1, 0), (0, 1)}
    assert out.proposals == 3


def test_deferred_acceptance_respects_mutual_acceptability_cutoff() -> None:
    # Only (c0,p0) is mutually acceptable; other edges fall below threshold on at least one side.
    v_c_true = (
        (0.9, 0.1),
        (0.1, 0.1),
    )
    v_p_true = (
        (0.9, 0.1),
        (0.1, 0.1),
    )
    v_c_hat = (
        (0.9, 0.8),
        (0.8, 0.9),
    )
    v_p_hat = (
        (0.9, 0.8),
        (0.8, 0.9),
    )

    out = deferred_acceptance(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.5,
        params=DeferredAcceptanceParams(proposer_side="customer"),
    )
    assert out.matches == ((0, 0),)


def test_deferred_acceptance_respects_truncated_receiver_list() -> None:
    v_c_true = (
        (0.9,),
        (0.9,),
    )
    v_p_true = ((0.9, 0.9),)
    v_c_hat = v_c_true
    v_p_hat = v_p_true

    out = deferred_acceptance(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.5,
        params=DeferredAcceptanceParams(
            proposer_side="customer", proposer_list_k=1, receiver_list_k=0
        ),
    )
    assert out.matches == ()
    assert out.proposals == 2
