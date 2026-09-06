"""Fit an initialized HalfKP output layer with WDL and legal-move ranking losses."""

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

from agent import evaluate
from tools.train_nnue import SCORE_SCALE, Example, forward, halfkp_indices


@dataclass(frozen=True, slots=True)
class RankedPosition:
    root: Example
    candidates: tuple[Example, ...]
    candidate_scores: tuple[float, ...]
    wdl_target: float
    validation: bool


def sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def make_example(board: chess.Board, score_cp: float, validation: bool) -> Example:
    baseline = float(evaluate(board, use_residual=False)) / SCORE_SCALE
    phase = min(
        24,
        sum(
            weight * len(board.pieces(piece_type, color))
            for piece_type, weight in (
                (chess.KNIGHT, 1),
                (chess.BISHOP, 1),
                (chess.ROOK, 2),
                (chess.QUEEN, 4),
            )
            for color in chess.COLORS
        ),
    )
    residual = score_cp / SCORE_SCALE - baseline
    return Example(
        halfkp_indices(board, board.turn),
        halfkp_indices(board, not board.turn),
        residual,
        residual,
        baseline,
        validation,
        phase,
        0,
    )


def game_target(result: str | None, turn: chess.Color) -> float | None:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else 0.0
    return None


def load_positions(path: Path, outcome_weight: float) -> list[RankedPosition]:
    positions: list[RankedPosition] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        candidates = record.get("candidates", ())
        if len(candidates) < 2:
            continue
        board = chess.Board(record["fen"])
        source = record.get("source")
        game_id = source.get("game_id") if isinstance(source, dict) else None
        split_key = str(game_id) if game_id else record["fen"]
        digest = hashlib.blake2b(split_key.encode(), digest_size=2).digest()
        validation = int.from_bytes(digest, "little") % 10 == 0
        teacher_score = float(record["score_cp"])
        teacher_wdl = sigmoid(teacher_score / SCORE_SCALE)
        result = source.get("result") if isinstance(source, dict) else None
        observed = game_target(result, board.turn)
        wdl_target = (
            teacher_wdl
            if observed is None
            else (1.0 - outcome_weight) * teacher_wdl + outcome_weight * observed
        )
        child_examples: list[Example] = []
        scores: list[float] = []
        for candidate in candidates:
            move = chess.Move.from_uci(candidate["move"])
            if move not in board.legal_moves:
                continue
            score = float(candidate["score_cp"])
            board.push(move)
            try:
                # UCI candidate scores use the root side's perspective; the child turns it around.
                child_examples.append(make_example(board, -score, validation))
                scores.append(score / SCORE_SCALE)
            finally:
                board.pop()
        if len(child_examples) >= 2:
            positions.append(
                RankedPosition(
                    make_example(board, teacher_score, validation),
                    tuple(child_examples),
                    tuple(scores),
                    wdl_target,
                    validation,
                )
            )
    return positions


def prediction(
    embedding: NDArray[np.float32],
    hidden_bias: NDArray[np.float32],
    output: NDArray[np.float32],
    output_bias: np.float32,
    example: Example,
    deployment_blend: float,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    residual, us, them = forward(embedding, hidden_bias, output, output_bias, example)
    return example.baseline + deployment_blend * residual, us, them


def metrics(
    embedding: NDArray[np.float32],
    hidden_bias: NDArray[np.float32],
    output: NDArray[np.float32],
    output_bias: np.float32,
    positions: list[RankedPosition],
    deployment_blend: float,
) -> tuple[float, float, float]:
    wdl_loss = 0.0
    brier = 0.0
    top1 = 0
    for position in positions:
        root_score, _, _ = prediction(
            embedding, hidden_bias, output, output_bias, position.root, deployment_blend
        )
        probability = sigmoid(root_score)
        target = position.wdl_target
        wdl_loss -= target * math.log(max(probability, 1e-9))
        wdl_loss -= (1.0 - target) * math.log(max(1.0 - probability, 1e-9))
        brier += (probability - target) ** 2
        move_values = [
            -prediction(
                embedding, hidden_bias, output, output_bias, child, deployment_blend
            )[0]
            for child in position.candidates
        ]
        top1 += int(int(np.argmax(move_values)) == int(np.argmax(position.candidate_scores)))
    count = max(1, len(positions))
    return wdl_loss / count, brier / count, top1 / count


def apply_gradient(
    embedding: NDArray[np.float32],
    hidden_bias: NDArray[np.float32],
    output: NDArray[np.float32],
    output_bias: np.float32,
    example: Example,
    derivative: float,
    rate: np.float32,
    fine_tune_hidden: bool,
) -> np.float32:
    """Apply one derivative with respect to the network's residual prediction."""
    _, us, them = forward(embedding, hidden_bias, output, output_bias, example)
    width = hidden_bias.size
    old_us_output = output[:width].copy()
    old_them_output = output[width : 2 * width].copy()
    step = np.float32(rate * derivative)
    output[:width] -= step * us
    output[width : 2 * width] -= step * them
    if not fine_tune_hidden:
        return output_bias
    us_gradient = derivative * old_us_output
    them_gradient = derivative * old_them_output
    us_active = (us > 0.0) & (us < 1.0)
    them_active = (them > 0.0) & (them < 1.0)
    embedding[example.us] -= rate * us_gradient * us_active
    embedding[example.them] -= rate * them_gradient * them_active
    hidden_bias -= rate * (us_gradient * us_active + them_gradient * them_active)
    return np.float32(output_bias - step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--wdl-weight", type=float, default=0.25)
    parser.add_argument("--outcome-weight", type=float, default=0.2)
    parser.add_argument("--deployment-blend", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fine-tune-hidden", action="store_true")
    args = parser.parse_args()

    positions = load_positions(args.dataset, args.outcome_weight)
    train = [position for position in positions if not position.validation]
    validation = [position for position in positions if position.validation]
    if not train or not validation:
        raise SystemExit("dataset must contain MultiPV positions in both splits")

    with np.load(args.initialize_from) as initial:
        embedding = initial["embedding"].astype(np.float32)
        hidden_bias = initial["hidden_bias"].astype(np.float32)
        output = initial["output"].astype(np.float32).reshape(-1)
        output_bias = np.float32(initial["output_bias"][0])

    rng = np.random.default_rng(args.seed)
    best_embedding = embedding.copy()
    best_hidden_bias = hidden_bias.copy()
    best_output = output.copy()
    best_output_bias = output_bias
    initial_metrics = metrics(
        embedding, hidden_bias, output, output_bias, validation, args.deployment_blend
    )
    print(
        f"initial validation logloss {initial_metrics[0]:.5f}, "
        f"brier {initial_metrics[1]:.5f}, top1 {initial_metrics[2]:.1%}"
    )
    best_metric = initial_metrics[0] + (1.0 - initial_metrics[2])
    order = np.arange(len(train))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        rate = np.float32(args.learning_rate / math.sqrt(epoch))
        for index in order:
            position = train[int(index)]
            root_score, _, _ = prediction(
                embedding,
                hidden_bias,
                output,
                output_bias,
                position.root,
                args.deployment_blend,
            )
            wdl_error = sigmoid(root_score) - position.wdl_target
            derivative = args.wdl_weight * args.deployment_blend * wdl_error
            output_bias = apply_gradient(
                embedding,
                hidden_bias,
                output,
                output_bias,
                position.root,
                derivative,
                rate,
                args.fine_tune_hidden,
            )

            best_child = position.candidates[0]
            best_score, _, _ = prediction(
                embedding,
                hidden_bias,
                output,
                output_bias,
                best_child,
                args.deployment_blend,
            )
            best_value = -best_score
            for child, teacher_value in zip(
                position.candidates[1:], position.candidate_scores[1:], strict=True
            ):
                child_score, _, _ = prediction(
                    embedding,
                    hidden_bias,
                    output,
                    output_bias,
                    child,
                    args.deployment_blend,
                )
                child_value = -child_score
                teacher_gap = position.candidate_scores[0] - teacher_value
                target = sigmoid(teacher_gap / 0.5)
                error = sigmoid((best_value - child_value) / 0.5) - target
                derivative = args.ranking_weight * args.deployment_blend * error / 0.5
                # Root move values negate the child-to-move evaluator.
                output_bias = apply_gradient(
                    embedding,
                    hidden_bias,
                    output,
                    output_bias,
                    best_child,
                    -derivative,
                    rate,
                    args.fine_tune_hidden,
                )
                output_bias = apply_gradient(
                    embedding,
                    hidden_bias,
                    output,
                    output_bias,
                    child,
                    derivative,
                    rate,
                    args.fine_tune_hidden,
                )

        current = metrics(
            embedding, hidden_bias, output, output_bias, validation, args.deployment_blend
        )
        selection = current[0] + (1.0 - current[2])
        print(
            f"epoch {epoch:2}: validation logloss {current[0]:.5f}, "
            f"brier {current[1]:.5f}, top1 {current[2]:.1%}"
        )
        if selection < best_metric:
            best_metric = selection
            best_embedding = embedding.copy()
            best_hidden_bias = hidden_bias.copy()
            best_output = output.copy()
            best_output_bias = output_bias

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embedding=best_embedding.astype(np.float16),
        hidden_bias=best_hidden_bias.astype(np.float16),
        output=best_output.astype(np.float16),
        output_bias=np.asarray([best_output_bias], dtype=np.float16),
        deployment_blend=np.asarray([args.deployment_blend], dtype=np.float16),
        dataset_sha256=np.asarray([hashlib.sha256(args.dataset.read_bytes()).hexdigest()]),
    )
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
