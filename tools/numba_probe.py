"""Measure whether a compiled bitboard evaluation core justifies Numba integration."""

from __future__ import annotations

import argparse
import time

import chess
import numpy as np
from numba import njit

from agent import EG_TABLE, EG_VALUE, MG_TABLE, MG_VALUE, PHASE_WEIGHT
from tools.benchmark import POSITIONS

MG_VALUE_ARRAY = np.asarray(MG_VALUE, dtype=np.int32)
EG_VALUE_ARRAY = np.asarray(EG_VALUE, dtype=np.int32)
PHASE_WEIGHT_ARRAY = np.asarray(PHASE_WEIGHT, dtype=np.int32)
MG_TABLE_ARRAY = np.asarray(MG_TABLE[1:], dtype=np.int32)
EG_TABLE_ARRAY = np.asarray(EG_TABLE[1:], dtype=np.int32)


@njit(cache=False)
def compiled_piece_score(bitboards: np.ndarray) -> tuple[int, int, int]:
    mg = 0
    eg = 0
    phase = 0
    for color_index in range(2):
        sign = -1 if color_index == 0 else 1
        for piece_index in range(6):
            pieces = bitboards[color_index * 6 + piece_index]
            piece_type = piece_index + 1
            for square in range(64):
                if pieces & (np.uint64(1) << np.uint64(square)):
                    index = square if color_index else square ^ 56
                    mg += sign * (
                        MG_VALUE_ARRAY[piece_type] + MG_TABLE_ARRAY[piece_index, index]
                    )
                    eg += sign * (
                        EG_VALUE_ARRAY[piece_type] + EG_TABLE_ARRAY[piece_index, index]
                    )
                    phase += PHASE_WEIGHT_ARRAY[piece_type]
    return mg, eg, phase


def encode(board: chess.Board, destination: np.ndarray) -> None:
    for color_index, color in enumerate((chess.BLACK, chess.WHITE)):
        for piece_index, piece_type in enumerate(range(chess.PAWN, chess.KING + 1)):
            destination[color_index * 6 + piece_index] = board.pieces_mask(piece_type, color)


def python_piece_score(board: chess.Board) -> tuple[int, int, int]:
    mg = eg = phase = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color else -1
        index = square if piece.color else chess.square_mirror(square)
        mg += sign * (MG_VALUE[piece.piece_type] + MG_TABLE[piece.piece_type][index])
        eg += sign * (EG_VALUE[piece.piece_type] + EG_TABLE[piece.piece_type][index])
        phase += PHASE_WEIGHT[piece.piece_type]
    return mg, eg, phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20_000)
    args = parser.parse_args()
    boards = [chess.Board(position.fen) for position in POSITIONS]
    buffer = np.empty(12, dtype=np.uint64)
    encode(boards[0], buffer)
    compiled_piece_score(buffer)  # Import-time warm-up shape used by a submitted engine.

    for board in boards:
        encode(board, buffer)
        if compiled_piece_score(buffer) != python_piece_score(board):
            raise AssertionError(f"compiled score mismatch for {board.fen()}")

    start = time.perf_counter()
    for index in range(args.repetitions):
        python_piece_score(boards[index % len(boards)])
    python_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for index in range(args.repetitions):
        encode(boards[index % len(boards)], buffer)
        compiled_piece_score(buffer)
    compiled_elapsed = time.perf_counter() - start

    print(f"python:   {args.repetitions / python_elapsed:,.0f} piece evaluations/s")
    print(f"compiled: {args.repetitions / compiled_elapsed:,.0f} piece evaluations/s")
    print(f"speedup:  {python_elapsed / compiled_elapsed:.2f}x including board encoding")


if __name__ == "__main__":
    main()
