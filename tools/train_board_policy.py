"""Train a compact board-conditioned legal-move policy with PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from torch import Tensor, nn

BOARD_FEATURES = 12 * 64
MOVE_FIELDS = 9


@dataclass(frozen=True, slots=True)
class Position:
    board: np.ndarray
    moves: np.ndarray
    scores: np.ndarray
    validation: bool


def board_features(board: chess.Board) -> np.ndarray:
    """Return side-relative, vertically oriented 12x64 binary planes."""
    result = np.zeros(BOARD_FEATURES, dtype=np.float32)
    for square, piece in board.piece_map().items():
        oriented = square if board.turn else square ^ 56
        relation = 0 if piece.color == board.turn else 1
        plane = relation * 6 + piece.piece_type - 1
        result[plane * 64 + oriented] = 1.0
    return result


def move_fields(board: chess.Board, move: chess.Move) -> tuple[int, ...]:
    piece = board.piece_type_at(move.from_square) or 0
    source = move.from_square if board.turn else move.from_square ^ 56
    target = move.to_square if board.turn else move.to_square ^ 56
    victim = board.piece_type_at(move.to_square) or 0
    if board.is_en_passant(move):
        victim = chess.PAWN
    return (
        piece,
        source,
        target,
        victim,
        move.promotion or 0,
        int(board.is_capture(move)),
        int(board.gives_check(move)),
        int(board.is_attacked_by(not board.turn, move.to_square)),
        int(board.is_attacked_by(board.turn, move.to_square)),
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
        moves: list[tuple[int, ...]] = []
        scores: list[float] = []
        for candidate in record.get("candidates", ()):
            move = chess.Move.from_uci(candidate["move"])
            if move in board.legal_moves:
                moves.append(move_fields(board, move))
                scores.append(float(candidate["score_cp"]))
        if len(moves) >= 2:
            positions.append(
                Position(
                    board_features(board),
                    np.asarray(moves, dtype=np.int64),
                    np.asarray(scores, dtype=np.float32),
                    validation,
                )
            )
    return positions


class BoardPolicy(nn.Module):
    """Bilinear board/move model small enough to distill or compile."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.board = nn.Sequential(nn.Linear(BOARD_FEATURES, width), nn.ReLU())
        cardinalities = (7, 64, 64, 7, 7, 2, 2, 2, 2)
        self.move_embeddings = nn.ModuleList(nn.Embedding(size, width) for size in cardinalities)
        self.move_biases = nn.ModuleList(nn.Embedding(size, 1) for size in cardinalities)

    def forward(self, board: Tensor, moves: Tensor) -> Tensor:
        context = self.board(board)
        move_vector = sum(
            embedding(moves[:, field]) for field, embedding in enumerate(self.move_embeddings)
        )
        bias = sum(
            embedding(moves[:, field]).squeeze(1)
            for field, embedding in enumerate(self.move_biases)
        )
        return (move_vector * context).sum(dim=1) / math.sqrt(context.shape[0]) + bias


@torch.no_grad()
def metrics(model: BoardPolicy, positions: list[Position]) -> tuple[float, float, float, float]:
    model.eval()
    correct = pairs = hard_correct = hard_pairs = top_one = 0
    regret = 0.0
    for position in positions:
        board = torch.from_numpy(position.board)
        moves = torch.from_numpy(position.moves)
        predicted = model(board, moves).numpy()
        selected = int(np.argmax(predicted))
        best = int(np.argmax(position.scores))
        top_one += int(selected == best)
        regret += float(position.scores[best] - position.scores[selected])
        order = np.argsort(position.scores)[::-1]
        ordered_predictions = predicted[order]
        ordered_scores = position.scores[order]
        pair_mask = np.triu(np.ones((len(order), len(order)), dtype=bool), k=1)
        wins = ordered_predictions[:, None] > ordered_predictions[None, :]
        gaps = ordered_scores[:, None] - ordered_scores[None, :]
        hard_mask = pair_mask & (gaps <= 100)
        correct += int(np.count_nonzero(wins & pair_mask))
        pairs += int(np.count_nonzero(pair_mask))
        hard_correct += int(np.count_nonzero(wins & hard_mask))
        hard_pairs += int(np.count_nonzero(hard_mask))
    count = max(1, len(positions))
    return (
        correct / max(1, pairs),
        hard_correct / max(1, hard_pairs),
        top_one / count,
        regret / count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--temperature-cp", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    positions = load_positions(args.dataset)
    train = [position for position in positions if not position.validation]
    validation = [position for position in positions if position.validation]
    if not train or not validation:
        raise SystemExit("dataset must contain train and validation positions")
    print(f"loaded {len(positions)} positions: {len(train)} train, {len(validation)} validation")

    model = BoardPolicy(args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state: dict[str, Tensor] | None = None
    best_hard = -1.0
    order = list(range(len(train)))
    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()
        total_loss = 0.0
        for index in order:
            position = train[index]
            board = torch.from_numpy(position.board)
            moves = torch.from_numpy(position.moves)
            target = torch.softmax(torch.from_numpy(position.scores) / args.temperature_cp, dim=0)
            prediction = model(board, moves)
            loss = -(target * torch.log_softmax(prediction, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        result = metrics(model, validation)
        print(
            f"epoch {epoch:2}: loss {total_loss / len(train):.4f}, "
            f"pair {result[0]:.1%}, hard-pair {result[1]:.1%}, "
            f"top-1 {result[2]:.1%}, regret {result[3]:.2f} cp"
        )
        if result[1] > best_hard:
            best_hard = result[1]
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: value.numpy() for name, value in best_state.items()}
    payload["dataset_sha256"] = np.asarray([hashlib.sha256(args.dataset.read_bytes()).hexdigest()])
    np.savez_compressed(args.output, **payload)
    model.load_state_dict(best_state)
    result = metrics(model, validation)
    print(
        f"best: pair {result[0]:.1%}, hard-pair {result[1]:.1%}, "
        f"top-1 {result[2]:.1%}, regret {result[3]:.2f} cp"
    )
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
