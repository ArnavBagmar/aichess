"""Generate ignored, offline-only Stockfish adapters for local strength calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = '''"""Generated offline calibration opponent; never package this directory."""

from __future__ import annotations

import chess
import chess.engine

ENGINE = chess.engine.SimpleEngine.popen_uci({executable!r})
ENGINE.configure({{"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": {elo}}})


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    think_s = min({think_s}, max(0.005, time_left_ms / 20_000.0))
    result = ENGINE.play(board, chess.engine.Limit(time=think_s))
    if result.move is None:
        raise ValueError("Stockfish returned no move")
    return result.move.uci()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-opponents"))
    parser.add_argument("--levels", type=int, nargs="+", default=[1320, 1500, 1700, 1900, 2000])
    parser.add_argument("--think-ms", type=int, default=100)
    args = parser.parse_args()

    executable = str(args.stockfish.resolve())
    for elo in args.levels:
        destination = args.output / f"stockfish-{elo}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "agent.py").write_text(
            TEMPLATE.format(executable=executable, elo=elo, think_s=args.think_ms / 1000),
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
