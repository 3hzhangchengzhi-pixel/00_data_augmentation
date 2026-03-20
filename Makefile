.PHONY: init test lint format lab clean

init:
	bash init.sh

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

lab:
	uv run jupyter lab

clean:
	rm -rf out/raw/* out/cleaned/* out/models/* out/reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
