"""Train a tiny phase-aware quiet-move ordering policy from MultiPV labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
from numpy.typing import NDArray

POLICY_PIECES = 7
POLICY_FROM_OFFSET = POLICY_PIECES
POLICY_TO_OFFSET = POLICY_FROM_OFFSET + 64
POLICY_PIECE_TO_OFFSET = POLICY_TO_OFFSET + 64
POLICY_BUCKET_SIZE = POLICY_PIECE_TO_OFFSET + POLICY_PIECES * 64
POLICY_BUCKETS = 3
POLICY_FEATURES = POLICY_BUCKETS * POLICY_BUCKET_SIZE


@dataclass(frozen=True, slots=True)
class Choice:
    features: tuple[int, int, int, int]
    score_cp: int


@dataclass(frozen=True, slots=True)
class Position:
    choices: tuple[Choice, ...]
    validation: bool


def sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def move_features(board: chess.Board, move: chess.Move) -> tuple[int, int, int, int]:
    """Return the four deployment indices for a quiet move."""
    pieces = board.occupied.bit_count()
    bucket = 0 if pieces <= 10 else 1 if pieces <= 22 else 2
    piece_type = board.piece_type_at(move.from_square) or 0
    from_square = move.from_square if board.turn else move.from_square ^ 56
    to_square = move.to_square if board.turn else move.to_square ^ 56
    offset = bucket * POLICY_BUCKET_SIZE
    return (
        offset + piece_type,
        offset + POLICY_FROM_OFFSET + from_square,
        offset + POLICY_TO_OFFSET + to_square,
        offset + POLICY_PIECE_TO_OFFSET + piece_type * 64 + to_square,
    )


def load_positions(path: Path) -> list[Position]:
    positions: list[Position] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        board = chess.Board(record["fen"])
        source = record.get("source")
        game_id = source.get("game_id") if isinstance(source, dict) else None
        split_key = str(game_id) if game_id else record["fen"]
        digest = hashlib.blake2b(split_key.encode(), digest_size=2).digest()
        validation = int.from_bytes(digest, "little") % 10 == 0
        choices: list[Choice] = []
        for candidate in record.get("candidates", ()):
            move = chess.Move.from_uci(candidate["move"])
            if (
                move in board.legal_moves
                and not board.is_capture(move)
                and move.promotion is None
            ):
                choices.append(
                    Choice(move_features(board, move), int(candidate["score_cp"]))
                )
        if len(choices) >= 2:
            choices.sort(key=lambda choice: choice.score_cp, reverse=True)
            positions.append(Position(tuple(choices), validation))
    return positions


def choice_score(weights: NDArray[np.float32], choice: Choice) -> float:
    return float(sum(weights[index] for index in choice.features))


def metrics(
    weights: NDArray[np.float32], positions: list[Position], temperature_cp: float
) -> tuple[float, float, float]:
    loss = 0.0
    correct = 0
    pairs = 0
    regret = 0.0
    for position in positions:
        predicted = [choice_score(weights, choice) for choice in position.choices]
        selected = int(np.argmax(predicted))
        regret += position.choices[0].score_cp - position.choices[selected].score_cp
        for left in range(len(position.choices)):
            for right in range(left + 1, len(position.choices)):
                target = sigmoid(
                    (position.choices[left].score_cp - position.choices[right].score_cp)
                    / temperature_cp
                )
                probability = sigmoid(predicted[left] - predicted[right])
                loss -= target * math.log(max(probability, 1e-9))
                loss -= (1.0 - target) * math.log(max(1.0 - probability, 1e-9))
                correct += int(predicted[left] > predicted[right])
                pairs += 1
    count = max(1, len(positions))
    return loss / max(1, pairs), correct / max(1, pairs), regret / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-nnue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature-cp", type=float, default=100.0)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--policy-scale", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    positions = load_positions(args.dataset)
    train = [position for position in positions if not position.validation]
    validation = [position for position in positions if position.validation]
    if not train or not validation:
        raise SystemExit("dataset must contain quiet MultiPV pairs in both splits")
    print(
        f"loaded {len(positions)} positions: {len(train)} train, "
        f"{len(validation)} validation"
    )

    weights = np.zeros(POLICY_FEATURES, dtype=np.float32)
    best_weights = weights.copy()
    best_loss = math.inf
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(train))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        rate = np.float32(args.learning_rate / math.sqrt(epoch))
        for index in order:
            position = train[int(index)]
            for left in range(len(position.choices)):
                for right in range(left + 1, len(position.choices)):
                    better = position.choices[left]
                    worse = position.choices[right]
                    target = sigmoid(
                        (better.score_cp - worse.score_cp) / args.temperature_cp
                    )
                    error = sigmoid(
                        choice_score(weights, better) - choice_score(weights, worse)
                    ) - target
                    for feature in better.features:
                        weights[feature] -= rate * (error + args.l2 * weights[feature])
                    for feature in worse.features:
                        weights[feature] -= rate * (-error + args.l2 * weights[feature])
        current = metrics(weights, validation, args.temperature_cp)
        print(
            f"epoch {epoch:2}: logloss {current[0]:.5f}, "
            f"pair accuracy {current[1]:.1%}, regret {current[2]:.2f} cp"
        )
        if current[0] < best_loss:
            best_loss = current[0]
            best_weights = weights.copy()

    quantized = np.rint(best_weights * args.policy_scale).astype(np.int16)
    with np.load(args.base_nnue) as base:
        artifact = {key: base[key] for key in base.files}
    artifact["policy_weights"] = quantized
    artifact["policy_scale"] = np.asarray([args.policy_scale], dtype=np.float32)
    artifact["policy_dataset_sha256"] = np.asarray(
        [hashlib.sha256(args.dataset.read_bytes()).hexdigest()]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **artifact)
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
