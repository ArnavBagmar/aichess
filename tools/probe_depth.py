"""Print fixed-depth root choices for one FEN without a clock cutoff."""

from __future__ import annotations

import argparse
import time

import chess

from agent import INFINITY, Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fen")
    parser.add_argument("--max-depth", type=int, default=10)
    args = parser.parse_args()

    board = chess.Board(args.fen)
    engine = Engine()
    engine.deadline = time.perf_counter() + 3_600
    for depth in range(1, args.max_depth + 1):
        engine.nodes = 0
        started = time.perf_counter()
        score, move = engine._root(board, depth, -INFINITY, INFINITY)
        elapsed = time.perf_counter() - started
        print(
            f"depth={depth} move={move.uci()} score={score} "
            f"nodes={engine.nodes} elapsed={elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
