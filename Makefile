SHELL := /bin/bash

.PHONY: setup play arena paired benchmark test zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

paired:
	uv run python -m tools.paired_arena --opponent baselines/greedy

benchmark:
	uv run python -m tools.benchmark

test:
	uv run python -m unittest -v

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m unittest -v
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
