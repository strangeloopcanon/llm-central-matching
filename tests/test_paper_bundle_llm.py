from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.paper_bundle_llm import main


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        if len(row) != len(header):
            raise ValueError("row length mismatch")
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_paper_bundle_llm_writes_outputs(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    ablations = tmp_path / "ablations"
    congestion = tmp_path / "congestion"
    heterogeneity = tmp_path / "heterogeneity"
    out_dir = tmp_path / "paper_llm"

    for d in (main_dir, ablations, congestion, heterogeneity):
        d.mkdir(parents=True, exist_ok=True)

    # Main 2x2 effects table (minimal).
    effects_header = [
        "category",
        "d_hat_I_ai_minus_standard",
        "d_hat_I_ai_minus_standard_se",
        "match_rate_did",
        "match_rate_did_se",
        "net_welfare_did",
        "net_welfare_did_se",
        "lambda_star_ai",
    ]
    _write_csv(
        main_dir / "effects_table.csv",
        effects_header,
        [
            ["easy", "0.001", "0.001", "0.000", "0.001", "0.000", "0.001", "0.0100"],
            ["hard", "0.006", "0.001", "0.003", "0.001", "0.002", "0.001", "0.0126"],
        ],
    )

    # Ablations parsing quality CSVs.
    pq_header = ["category", "mode", "side", "mean_l1", "top1_accuracy"]
    for category in ("easy", "hard"):
        rows: list[list[str]] = []
        for mode in ("form_top3", "free_text_gpt", "chat_gpt"):
            for side in ("customer", "provider"):
                rows.append([category, mode, side, "0.1234", "0.500"])
        _write_csv(ablations / f"parsing_quality_{category}.csv", pq_header, rows)

    # Congestion saturation CSVs for 0% and 100%.
    cong_header = [
        "category",
        "saturation",
        "net_welfare_per_customer",
        "inbox_per_provider_per_day",
    ]
    for category in ("easy", "hard"):
        _write_csv(
            congestion / f"congestion_saturation_{category}.csv",
            cong_header,
            [
                [category, "0.0", "0.10", "1.0"],
                [category, "1.0", "-0.05", "10.0"],
            ],
        )

    # Heterogeneity LLM sweep runs.csv
    het_header = [
        "utility_noise_sd",
        "category",
        "net_welfare_did",
        "net_welfare_did_se",
        "ai_central_mean",
        "std_central_mean",
    ]
    _write_csv(
        heterogeneity / "runs.csv",
        het_header,
        [
            ["0.08", "hard", "0.0010", "0.0020", "0.10", "0.09"],
            ["0.15", "hard", "0.0005", "0.0025", "0.08", "0.08"],
        ],
    )

    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(out_dir),
            "--main-dir",
            str(main_dir),
            "--ablations-dir",
            str(ablations),
            "--congestion-dir",
            str(congestion),
            "--heterogeneity-dir",
            str(heterogeneity),
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (out_dir / "README.md").exists()
    assert (out_dir / "key_results.csv").exists()
    assert (out_dir / "summary.json").exists()
