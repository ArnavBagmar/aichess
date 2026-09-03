"""Train a compact tapered linear chess evaluator from JSONL teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import chess

from agent import evaluate

FEATURES = 2 * 6 * 64
MAX_PHASE = 24
PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)


@dataclass(frozen=True, slots=True)
class Example:
    active: tuple[tuple[int, float], ...]
    target: float
    baseline: float
    split: int


def features(board: chess.Board) -> tuple[tuple[int, float], ...]:
    phase = min(
        MAX_PHASE,
        sum(PHASE_WEIGHT[piece.piece_type] for piece in board.piece_map().values()),
    )
    mg_scale = phase / MAX_PHASE
    eg_scale = 1.0 - mg_scale
    active: list[tuple[int, float]] = []
    for square, piece in board.piece_map().items():
        oriented = square if piece.color else chess.square_mirror(square)
        base = (piece.piece_type - 1) * 64 + oriented
        sign = 1.0 if piece.color else -1.0
        active.append((base, sign * mg_scale))
        active.append((384 + base, sign * eg_scale))
    return tuple(active)


def load_examples(path: Path, score_limit: int) -> list[Example]:
    examples: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        board = chess.Board(record["fen"])
        score = max(-score_limit, min(score_limit, int(record["score_cp"])))
        white_score = score if board.turn else -score
        digest = hashlib.blake2b(record["fen"].encode(), digest_size=2).digest()
        split = int.from_bytes(digest, "little") % 10
        baseline = evaluate(board) * (1.0 if board.turn else -1.0)
        examples.append(Example(features(board), float(white_score), baseline, split))
        for candidate in record.get("candidates", []):
            move = chess.Move.from_uci(candidate["move"])
            if move not in board.legal_moves:
                continue
            candidate_score = max(
                -score_limit, min(score_limit, int(candidate["score_cp"]))
            )
            candidate_white_score = candidate_score if board.turn else -candidate_score
            board.push(move)
            try:
                candidate_baseline = evaluate(board) * (1.0 if board.turn else -1.0)
                examples.append(
                    Example(
                        features(board),
                        float(candidate_white_score),
                        candidate_baseline,
                        split,
                    )
                )
            finally:
                board.pop()
    return examples


def predict(weights: list[float], example: Example) -> float:
    return example.baseline + sum(weights[index] * value for index, value in example.active)


def metrics(weights: list[float], examples: list[Example]) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0
    errors = [predict(weights, example) - example.target for example in examples]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return mae, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/evaluator.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--huber", type=float, default=200.0)
    parser.add_argument("--score-limit", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    examples = load_examples(args.dataset, args.score_limit)
    train = [example for example in examples if example.split != 0]
    validation = [example for example in examples if example.split == 0]
    if not train or not validation:
        raise SystemExit("dataset must produce non-empty deterministic train and validation splits")

    rng = random.Random(args.seed)
    weights = [0.0] * FEATURES
    best_weights = weights.copy()
    best_validation = metrics(weights, validation)[0]
    stale_epochs = 0
    print(f"baseline validation mae {best_validation:.2f}")
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        rate = args.learning_rate / math.sqrt(epoch)
        for example in train:
            error = predict(weights, example) - example.target
            gradient = max(-args.huber, min(args.huber, error))
            for index, value in example.active:
                weights[index] -= rate * gradient * value / 32.0
        train_mae, _ = metrics(weights, train)
        validation_mae, validation_rmse = metrics(weights, validation)
        print(
            f"epoch {epoch:3}: train mae {train_mae:7.2f}, "
            f"validation mae {validation_mae:7.2f}, rmse {validation_rmse:7.2f}"
        )
        if validation_mae < best_validation:
            best_validation = validation_mae
            best_weights = weights.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"early stopping; restoring validation-best epoch at "
                    f"{best_validation:.2f} mae"
                )
                break

    quantized = [round(value) for value in best_weights]
    validation_mae, validation_rmse = metrics([float(value) for value in quantized], validation)
    dataset_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    artifact = {
        "format": 1,
        "architecture": "handcrafted-plus-tapered-linear-residual-768",
        "weights": quantized,
        "provenance": {
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_sha,
            "examples": len(examples),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "huber": args.huber,
            "seed": args.seed,
            "patience": args.patience,
            "validation_mae_cp": round(validation_mae, 3),
            "validation_rmse_cp": round(validation_rmse, 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
