install: pyproject.toml uv.lock
	uv sync

run:
	uv run python -m src.main

# Trzeba zrobić jeszcze debug
debug:

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	flake8 --exclude .venv,llm_sdk .
	mypy . --exclude 'llm_sdk|\.venv' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 --exclude .venv,llm_sdk .
	mypy . --exclude 'llm_sdk|\.venv' --strict
