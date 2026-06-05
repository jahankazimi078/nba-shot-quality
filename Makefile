PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
STATIC_PORT ?= 8000

.PHONY: setup test app-artifacts app

setup:
	python3 -m venv .venv
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m ruff check .
	$(PY) -m pytest

app-artifacts:
	$(PY) -m nba_shot_quality.cli app-artifacts --seasons 2022-23 2023-24 2024-25

app:
	$(PY) -m http.server $(STATIC_PORT) -d static
