.PHONY: setup format check test llm-live all experiment regime congestion field field-v2 field-v2-sensitivity ablations intakes calibrate field-v2-calibrated paper-bundle clean

setup:
	uv venv
	uv pip install --python .venv/bin/python -e ".[dev]"

format:
	uv run ruff format .
	uv run ruff check --fix .

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run bandit -q -r src
	uv run detect-secrets-hook --baseline .secrets.baseline --exclude-files '(^|/)\.env$$'

test:
	uv run python -m pytest --cov

llm-live:
	uv run python -m pytest -q tests_llm_live

all: check test llm-live

experiment:
	uv run python -m econ_llm_preferences_experiment.run --out reports/latest

regime:
	uv run python -m econ_llm_preferences_experiment.regime_sweep --out reports/latest --category hard

congestion:
	uv run python -m econ_llm_preferences_experiment.experiment2_congestion --out reports/latest_congestion

field:
	uv run python -m econ_llm_preferences_experiment.publishable_field_sim --out reports/field_latest

field-v2:
	uv run python -m econ_llm_preferences_experiment.field_sim_v2 --out reports/field_v2_latest

field-v2-sensitivity:
	uv run python -m econ_llm_preferences_experiment.field_sim_v2_sensitivity --out reports/field_v2_sensitivity_latest --central-rec-k "3,5,7"

ablations:
	uv run python -m econ_llm_preferences_experiment.llm_ablations --out reports/ablations_latest

intakes:
	uv run python -m econ_llm_preferences_experiment.field_sim_v2_intakes --out reports/intakes_latest

calibrate:
	uv run python -m econ_llm_preferences_experiment.field_sim_v2_calibrate --out reports/field_v2_calibration_latest

field-v2-calibrated: calibrate
	uv run python -m econ_llm_preferences_experiment.field_sim_v2 --out reports/field_v2_calibrated_latest --params-json reports/field_v2_calibration_latest/best_params.json

paper-bundle:
	uv run python -m econ_llm_preferences_experiment.paper_bundle --out reports/paper_latest

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache reports/latest reports/latest_congestion reports/field_latest reports/field_v2_latest reports/field_v2_sensitivity_latest reports/ablations_latest reports/intakes_latest reports/field_v2_calibration_latest reports/field_v2_calibrated_latest reports/paper_latest
