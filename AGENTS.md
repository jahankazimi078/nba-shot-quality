# Repository Guidelines

## Project Structure & Module Organization

This Python package uses a `src/` layout. Core code lives in `src/nba_shot_quality/`: `ingest/` fetches NBA data, `features/` builds shot and lineup features, `models/` contains xPoints, POE, and RAPM logic, `eval/` holds diagnostics, and `analysis/` contains coaching-change assets. The CLI entry point is `src/nba_shot_quality/cli.py`. The Streamlit dashboard is in `app/streamlit_app.py`, and reusable pipeline launchers are in `scripts/`. Runtime outputs belong under `data/` and `reports/`; keep generated artifacts out of source modules.

## Build, Test, and Development Commands

- `python3 -m venv .venv`: create a local virtual environment.
- `.venv/bin/pip install -e .`: install the package in editable mode.
- `.venv/bin/pip install -r requirements.txt`: install pinned runtime dependencies.
- `.venv/bin/python -m nba_shot_quality.cli ingest --season 2024-25`: fetch regular-season shot data.
- `.venv/bin/python -m nba_shot_quality.cli features --season 2024-25`: build processed shot features.
- `bash scripts/run_xpoints.sh 2024-25`: run the xPoints pipeline.
- `bash scripts/run_poe.sh 2024-25 2023-24`: run POE scoring and stability analysis.
- `.venv/bin/streamlit run app/streamlit_app.py`: launch the dashboard.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax and follow the existing straightforward module style. Prefer 4-space indentation, snake_case for functions, variables, and module names, and PascalCase only for classes. Keep CLI subcommands short and action-oriented, such as `ingest`, `features`, `train`, or `poe-vs-rts`. No formatter is configured; before committing, review diffs for readable imports and clear names.

## Testing Guidelines

There is currently no dedicated `tests/` directory or configured test runner. For pipeline changes, validate with the narrowest relevant CLI command first, then run the affected script when feasible. Add future tests under `tests/` using names like `test_shot_features.py`, and prefer small fixtures over live `nba_api` calls. For ingestion changes, preserve cache behavior and document network-dependent checks.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects, for example `Add RAPM diagnostics, offense/net ratings, and PBP validation check`. Follow that pattern: start with a verb, name the feature or fix, and keep the subject specific. Pull requests should include a short summary, commands run, affected seasons or data files, and screenshots for dashboard changes. Link related issues when available and call out skipped validation.

## Security & Configuration Tips

Do not commit virtual environments, local caches, or generated parquet/report outputs unless explicitly required. Avoid embedding credentials or private endpoints; this project should run from pinned dependencies plus public NBA data sources.
