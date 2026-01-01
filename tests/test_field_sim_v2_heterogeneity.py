from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2_heterogeneity import main


def test_field_sim_v2_heterogeneity_writes_outputs(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed-base",
            "7",
            "--n-seeds",
            "1",
            "--cities",
            "4",
            "--weeks",
            "1",
            "--jobs-easy",
            "3",
            "--jobs-hard",
            "3",
            "--providers",
            "6",
            "--weight-alphas",
            "1.0",
            "--ai-noise-sds",
            "0.03",
            "--misclass-rates",
            "0.7",
            "--utility-noise-sds",
            "0.08",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "fig_triple_vs_weight_alpha.svg").exists()
