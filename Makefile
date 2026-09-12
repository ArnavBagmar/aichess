SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

# Placeholder net until Phase 2 exports a trained one. Every target that imports
# agent.py (or packages it) depends on this so a fresh clone never ships or plays
# without weights.
weights/nnue.npz:
	uv run python tools/gen_random_net.py

play: weights/nnue.npz
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena: weights/nnue.npz
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip: weights/nnue.npz
	uv run python -m harness.package

gate: weights/nnue.npz
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
