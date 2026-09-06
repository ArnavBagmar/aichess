"""Train a compact legal-move ranking policy from complete root-move labels."""

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

POLICY_FEATURES = 1 << 16


@dataclass(frozen=True, slots=True)
class Choice:
    features: tuple[int, ...]
    score_cp: int


@dataclass(frozen=True, slots=True)
class Position:
    choices: tuple[Choice, ...]
    validation: bool


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def feature_index(namespace: int, *values: int) -> int:
    """Return a stable index independent of Python's randomized hash."""
    value = (namespace + 1) * 0x9E3779B1
    for item in values:
        value ^= (item + 1) * 0x85EBCA77
        value = ((value << 13) | (value >> 19)) & 0xFFFFFFFF
        value = (value * 0xC2B2AE3D) & 0xFFFFFFFF
    return value & (POLICY_FEATURES - 1)


def move_features(board: chess.Board, move: chess.Move) -> tuple[int, ...]:
    """Encode geometry, tactics, and context for any legal move."""
    phase = 0 if len(board.piece_map()) <= 10 else 1 if len(board.piece_map()) <= 22 else 2
    piece = board.piece_type_at(move.from_square) or 0
    source = move.from_square if board.turn else move.from_square ^ 56
    target = move.to_square if board.turn else move.to_square ^ 56
    victim = board.piece_type_at(move.to_square) or 0
    if board.is_en_passant(move):
        victim = chess.PAWN
    promotion = move.promotion or 0
    capture = int(board.is_capture(move))
    check = int(board.gives_check(move))
    attacked = int(board.is_attacked_by(not board.turn, move.to_square))
    defended = int(board.is_attacked_by(board.turn, move.to_square))
    rank_delta = chess.square_rank(target) - chess.square_rank(source)
    file_delta = chess.square_file(target) - chess.square_file(source)
    return (
        feature_index(0, phase, piece),
        feature_index(1, phase, piece, source),
        feature_index(2, phase, piece, target),
        feature_index(3, phase, piece, source, target),
        feature_index(4, phase, source, target),
        feature_index(5, phase, piece, capture, victim),
        feature_index(6, phase, piece, capture, victim, target),
        feature_index(7, phase, piece, promotion),
        feature_index(8, phase, piece, check),
        feature_index(9, phase, piece, attacked, defended),
        feature_index(10, phase, piece, rank_delta, file_delta),
        feature_index(11, phase, piece, target, attacked, defended),
    )


def load_positions(path: Path) -> list[Position]:
    positions: list[Position] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        board = chess.Board(record["fen"])
        source_record = record.get("source")
        game_id = source_record.get("game_id") if isinstance(source_record, dict) else None
        split_key = str(game_id) if game_id else record["fen"]
        validation = (
            int.from_bytes(hashlib.blake2b(split_key.encode(), digest_size=2).digest(), "little")
            % 10
            == 0
        )
        choices = []
        for candidate in record.get("candidates", ()):
            move = chess.Move.from_uci(candidate["move"])
            if move in board.legal_moves:
                choices.append(Choice(move_features(board, move), int(candidate["score_cp"])))
        if len(choices) >= 2:
            choices.sort(key=lambda choice: choice.score_cp, reverse=True)
            positions.append(Position(tuple(choices), validation))
    return positions


def choice_score(weights: NDArray[np.float32], choice: Choice) -> float:
    return float(sum(weights[index] for index in choice.features))


def training_pairs(position: Position) -> list[tuple[int, int]]:
    """Use linear-many informative pairs instead of quadratic redundant pairs."""
    pairs = {(0, right) for right in range(1, len(position.choices))}
    pairs.update((left, left + 1) for left in range(1, len(position.choices) - 1))
    return sorted(pairs)


def metrics(
    weights: NDArray[np.float32], positions: list[Position], temperature_cp: float
) -> tuple[float, float, float, float, float, float]:
    loss = regret = 0.0
    correct = pairs = hard_correct = hard_pairs = top_one = top_three = 0
    for position in positions:
        predicted = [choice_score(weights, choice) for choice in position.choices]
        selected = int(np.argmax(predicted))
        regret += position.choices[0].score_cp - position.choices[selected].score_cp
        top_one += int(selected == 0)
        top_three += int(selected < 3)
        for left in range(len(position.choices)):
            for right in range(left + 1, len(position.choices)):
                target = sigmoid(
                    (position.choices[left].score_cp - position.choices[right].score_cp)
                    / temperature_cp
                )
                probability = sigmoid(predicted[left] - predicted[right])
                loss -= target * math.log(max(probability, 1e-9)) + (1.0 - target) * math.log(
                    max(1.0 - probability, 1e-9)
                )
                correct += int(predicted[left] > predicted[right])
                pairs += 1
                if position.choices[left].score_cp - position.choices[right].score_cp <= 100:
                    hard_correct += int(predicted[left] > predicted[right])
                    hard_pairs += 1
    count = max(1, len(positions))
    return (
        loss / max(1, pairs),
        correct / max(1, pairs),
        hard_correct / max(1, hard_pairs),
        top_one / count,
        top_three / count,
        regret / count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-nnue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature-cp", type=float, default=100.0)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--policy-scale", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    positions = load_positions(args.dataset)
    train = [position for position in positions if not position.validation]
    validation = [position for position in positions if position.validation]
    if not train or not validation:
        raise SystemExit("dataset must contain legal-move pairs in both splits")
    print(f"loaded {len(positions)} positions: {len(train)} train, {len(validation)} validation")
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
            for left, right in training_pairs(position):
                better, worse = position.choices[left], position.choices[right]
                target = sigmoid((better.score_cp - worse.score_cp) / args.temperature_cp)
                error = (
                    sigmoid(choice_score(weights, better) - choice_score(weights, worse)) - target
                )
                for feature in better.features:
                    weights[feature] -= rate * (error + args.l2 * weights[feature])
                for feature in worse.features:
                    weights[feature] -= rate * (-error + args.l2 * weights[feature])
        current = metrics(weights, validation, args.temperature_cp)
        print(
            f"epoch {epoch:2}: logloss {current[0]:.5f}, "
            f"pair accuracy {current[1]:.1%}, hard-pair {current[2]:.1%}, "
            f"top-1 {current[3]:.1%}, top-3 {current[4]:.1%}, "
            f"regret {current[5]:.2f} cp"
        )
        if current[0] < best_loss:
            best_loss, best_weights = current[0], weights.copy()

    quantized = np.rint(best_weights * args.policy_scale).astype(np.int16)
    with np.load(args.base_nnue) as base:
        artifact = {key: base[key] for key in base.files}
    artifact["policy_v2_weights"] = quantized
    artifact["policy_v2_scale"] = np.asarray([args.policy_scale], dtype=np.float32)
    artifact["policy_v2_dataset_sha256"] = np.asarray(
        [hashlib.sha256(args.dataset.read_bytes()).hexdigest()]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **artifact)
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
