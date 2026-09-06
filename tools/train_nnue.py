"""Train a compact two-perspective HalfKP value network with NumPy."""

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

PIECE_TYPES = 5
FEATURES = 64 * 2 * PIECE_TYPES * 64
SCORE_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class Example:
    us: NDArray[np.int32]
    them: NDArray[np.int32]
    target: float
    teacher_residual: float
    baseline: float
    validation: bool
    phase: int
    phase_bucket: int


def halfkp_indices(board: chess.Board, perspective: chess.Color) -> NDArray[np.int32]:
    king = board.king(perspective)
    if king is None:
        raise ValueError("HalfKP requires both kings")
    oriented_king = king if perspective else chess.square_mirror(king)
    indices: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        oriented_square = square if perspective else chess.square_mirror(square)
        relation = 0 if piece.color == perspective else 1
        piece_index = relation * PIECE_TYPES + piece.piece_type - 1
        indices.append((oriented_king * 10 + piece_index) * 64 + oriented_square)
    return np.asarray(indices, dtype=np.int32)


def load_examples(
    path: Path,
    deployment_blend: float = 1.0,
    split_by_game: bool = True,
    phase_buckets: int = 1,
) -> list[Example]:
    examples: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        board = chess.Board(record["fen"])
        side = board.turn
        source = record.get("source")
        game_id = source.get("game_id") if isinstance(source, dict) else None
        split_key = str(game_id) if split_by_game and game_id else record["fen"]
        digest = hashlib.blake2b(split_key.encode(), digest_size=2).digest()
        validation = int.from_bytes(digest, "little") % 10 == 0
        baseline = float(evaluate(board, use_residual=False)) / SCORE_SCALE
        teacher_residual = float(record["score_cp"]) / SCORE_SCALE - baseline
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
        phase_bucket = min(phase_buckets - 1, phase * phase_buckets // 25)
        examples.append(
            Example(
                halfkp_indices(board, side),
                halfkp_indices(board, not side),
                teacher_residual / deployment_blend,
                teacher_residual,
                baseline,
                validation,
                phase,
                phase_bucket,
            )
        )
    return examples


def forward(
    embedding: NDArray[np.float32],
    hidden_bias: NDArray[np.float32],
    output: NDArray[np.float32],
    output_bias: np.float32,
    example: Example,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    width = hidden_bias.shape[0]
    offset = example.phase_bucket * width * 2
    us = np.clip(hidden_bias + embedding[example.us].sum(axis=0), 0.0, 1.0)
    them = np.clip(hidden_bias + embedding[example.them].sum(axis=0), 0.0, 1.0)
    prediction = float(
        us @ output[offset : offset + width]
        + them @ output[offset + width : offset + 2 * width]
        + output_bias
    )
    return prediction, us, them


def metrics(
    embedding: NDArray[np.float32],
    hidden_bias: NDArray[np.float32],
    output: NDArray[np.float32],
    output_bias: np.float32,
    examples: list[Example],
    deployment_blend: float = 1.0,
) -> tuple[float, float, float]:
    errors: list[float] = []
    baseline_errors: list[float] = []
    for example in examples:
        prediction, _, _ = forward(embedding, hidden_bias, output, output_bias, example)
        errors.append(
            (deployment_blend * prediction - example.teacher_residual) * SCORE_SCALE
        )
        baseline_errors.append(-example.teacher_residual * SCORE_SCALE)
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    baseline_mae = sum(abs(error) for error in baseline_errors) / len(baseline_errors)
    return mae, rmse, baseline_mae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/nnue.npz"))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--phase-buckets", type=int, default=1)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="initialize float weights from an existing model",
    )
    parser.add_argument(
        "--freeze-hidden",
        action="store_true",
        help="train only phase-specific output heads and keep the shared feature transformer fixed",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        help="train only examples whose Stockfish phase value is at most this value",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--deployment-blend",
        type=float,
        default=1.0,
        help="fraction of the residual applied by the deployed engine",
    )
    parser.add_argument(
        "--split-by",
        choices=("game", "position"),
        default="game",
        help="keep positions from one source game in the same data split",
    )
    args = parser.parse_args()

    if not 0.0 < args.deployment_blend <= 1.0:
        raise SystemExit("deployment blend must be in (0, 1]")
    if args.phase_buckets < 1:
        raise SystemExit("phase buckets must be positive")
    examples = load_examples(
        args.dataset,
        args.deployment_blend,
        split_by_game=args.split_by == "game",
        phase_buckets=args.phase_buckets,
    )
    if args.max_phase is not None:
        if not 0 <= args.max_phase <= 24:
            raise SystemExit("max phase must be between 0 and 24")
        examples = [example for example in examples if example.phase <= args.max_phase]
    train = [example for example in examples if not example.validation]
    validation = [example for example in examples if example.validation]
    if not train or not validation:
        raise SystemExit("dataset must produce non-empty train and validation splits")

    rng = np.random.default_rng(args.seed)
    if args.initialize_from:
        with np.load(args.initialize_from) as initial:
            embedding = initial["embedding"].astype(np.float32)
            hidden_bias = initial["hidden_bias"].astype(np.float32)
            base_output = initial["output"].astype(np.float32).reshape(-1)
            initial_buckets = base_output.size // (2 * hidden_bias.size)
            if hidden_bias.size != args.hidden:
                raise SystemExit("initialized hidden width does not match --hidden")
            if initial_buckets == args.phase_buckets:
                output = base_output.copy()
            elif initial_buckets == 1:
                output = np.tile(base_output, args.phase_buckets)
            else:
                raise SystemExit("initialized phase heads cannot be mapped to requested buckets")
            bias_key = "output_bias" if "output_bias" in initial else "output_bias_cp"
            output_bias = np.float32(initial[bias_key][0])
            if bias_key == "output_bias_cp":
                output_bias /= np.float32(SCORE_SCALE)
    else:
        embedding = rng.normal(0.0, 0.01, (FEATURES, args.hidden)).astype(np.float32)
        hidden_bias = np.full(args.hidden, 0.1, dtype=np.float32)
        output = rng.normal(0.0, 0.02, args.hidden * 2 * args.phase_buckets).astype(np.float32)
        output_bias = np.float32(0.0)
    best_mae = math.inf
    best: (
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], np.float32]
        | None
    ) = None

    order = np.arange(len(train))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        rate = np.float32(args.learning_rate / math.sqrt(epoch))
        for index in order:
            example = train[int(index)]
            prediction, us, them = forward(
                embedding, hidden_bias, output, output_bias, example
            )
            error = np.float32(np.clip(prediction - example.target, -2.5, 2.5))
            old_output = output.copy()
            offset = example.phase_bucket * args.hidden * 2
            output[offset : offset + args.hidden] -= rate * error * us
            output[offset + args.hidden : offset + 2 * args.hidden] -= rate * error * them
            if not args.freeze_hidden:
                output_bias -= rate * error
            us_gradient = error * old_output[offset : offset + args.hidden]
            them_gradient = error * old_output[
                offset + args.hidden : offset + 2 * args.hidden
            ]
            us_active = (us > 0.0) & (us < 1.0)
            them_active = (them > 0.0) & (them < 1.0)
            if not args.freeze_hidden:
                embedding[example.us] -= rate * us_gradient * us_active
                embedding[example.them] -= rate * them_gradient * them_active
                hidden_bias -= rate * (us_gradient * us_active + them_gradient * them_active)

        mae, rmse, baseline_mae = metrics(
            embedding,
            hidden_bias,
            output,
            output_bias,
            validation,
            args.deployment_blend,
        )
        print(
            f"epoch {epoch:2}: validation mae {mae:7.2f}, rmse {rmse:7.2f}, "
            f"handcrafted mae {baseline_mae:7.2f}"
        )
        if mae < best_mae:
            best_mae = mae
            best = (embedding.copy(), hidden_bias.copy(), output.copy(), output_bias)

    if best is None:
        raise RuntimeError("training produced no model")
    embedding, hidden_bias, output, output_bias = best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embedding=embedding.astype(np.float16),
        hidden_bias=hidden_bias.astype(np.float16),
        output=output.astype(np.float16),
        output_bias=np.asarray([output_bias], dtype=np.float16),
        deployment_blend=np.asarray([args.deployment_blend], dtype=np.float16),
        phase_buckets=np.asarray([args.phase_buckets], dtype=np.int16),
        dataset_sha256=np.asarray([hashlib.sha256(args.dataset.read_bytes()).hexdigest()]),
    )
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes), best mae {best_mae:.2f}")


if __name__ == "__main__":
    main()
