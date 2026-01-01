from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.paper_bundle import main


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        if len(row) != len(header):
            raise ValueError("row length mismatch")
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_paper_bundle_writes_outputs(tmp_path: Path) -> None:
    field_v2 = tmp_path / "field_v2"
    sens = tmp_path / "sens"
    ablations = tmp_path / "ablations"
    congestion = tmp_path / "congestion"
    minimal = tmp_path / "minimal"
    intakes = tmp_path / "intakes"
    out_dir = tmp_path / "paper"

    for d in (field_v2, sens, ablations, congestion, minimal, intakes):
        d.mkdir(parents=True, exist_ok=True)

    # FieldSim v2 regression CSVs: minimal rows required by paper_bundle.py
    reg_header = ["term", "coef", "se(cluster)", "t", "p(normal)"]
    reg_rows = [
        ["ai_x_central_x_hard", "0.10", "0.01", "10.0", "0.0000"],
        ["n_obs / n_clusters", "100 / 4", "", "", ""],
    ]
    for outcome in ("matched", "canceled", "consumer_surplus", "provider_profit", "net_welfare"):
        _write_csv(field_v2 / f"reg_{outcome}.csv", reg_header, reg_rows)

    # Sensitivity summary JSON.
    (sens / "summary.json").write_text(
        '{"n_runs": 1, "pos_share_triple_welfare": 1.0, "sig_share_triple_welfare_p05": 0.0}\n',
        encoding="utf-8",
    )

    # Ablations parsing quality CSVs (easy/hard) for required (mode,side) rows.
    pq_header = ["category", "mode", "side", "mean_l1", "top1_accuracy"]
    for category in ("easy", "hard"):
        rows: list[list[str]] = []
        for mode in ("form_top3", "free_text_gpt", "chat_gpt"):
            for side in ("customer", "provider"):
                rows.append([category, mode, side, "0.1234", "0.500"])
        _write_csv(ablations / f"parsing_quality_{category}.csv", pq_header, rows)

    # Congestion saturation CSVs (easy/hard) with required saturations.
    cong_header = [
        "category",
        "saturation",
        "net_welfare_per_customer",
        "match_rate_treated",
        "match_rate_control",
        "inbox_per_provider_per_day",
        "provider_response_rate",
    ]
    for category in ("easy", "hard"):
        _write_csv(
            congestion / f"congestion_saturation_{category}.csv",
            cong_header,
            [
                [category, "0.0", "0.10", "0.10", "0.10", "1.0", "0.50"],
                [category, "0.25", "0.08", "0.12", "0.09", "2.0", "0.40"],
                [category, "1.0", "-0.05", "0.02", "0.03", "10.0", "0.02"],
            ],
        )

    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(out_dir),
            "--field-v2-dir",
            str(field_v2),
            "--sens-dir",
            str(sens),
            "--ablations-dir",
            str(ablations),
            "--congestion-dir",
            str(congestion),
            "--minimal-dir",
            str(minimal),
            "--intakes-dir",
            str(intakes),
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (out_dir / "README.md").exists()
    assert (out_dir / "key_results.csv").exists()
    assert (out_dir / "key_results.md").exists()
