.PHONY: setup setup-maniskill setup-mujoco test lint validate smoke

setup:
	uv sync --extra dev --extra mujoco --extra video --extra reports

setup-maniskill:
	uv sync --extra dev --extra maniskill --extra video --extra reports

setup-mujoco:
	uv sync --extra dev --extra mujoco --extra video --extra reports

test:
	uv run pytest -q

lint:
	uv run ruff check .

validate:
	uv run python scripts/validate_configs.py

smoke:
	uv run python scripts/release_smoke.py
