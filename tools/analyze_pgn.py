"""Find the largest teacher-evaluation drops after one side's moves in a PGN."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.pgn

from tools.label_positions import UciTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--color", choices=("white", "black"), required=True)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()

    with args.pgn.open(encoding="utf-8") as source:
        game = chess.pgn.read_game(source)
    if game is None:
        raise SystemExit("PGN contains no game")
    agent_color = chess.WHITE if args.color == "white" else chess.BLACK
    board = game.board()
    drops: list[tuple[int, int, str, str, int]] = []
    with UciTeacher(args.stockfish.resolve(), 64, 1) as teacher:
        initial = teacher.analyze(board, args.depth)[0]
        for ply, move in enumerate(game.mainline_moves(), 1):
            mover = board.turn
            before_analysis = (
                teacher.analyze(board, args.depth) if mover == agent_color else (0, "-", 0, [])
            )
            board.push(move)
            if mover != agent_color or board.is_game_over():
                continue
            teacher_score = teacher.analyze(board, args.depth)[0]
            agent_score = -teacher_score
            drops.append(
                (before_analysis[0] - agent_score, ply, move.uci(), before_analysis[1], agent_score)
            )

    print(f"initial agent score: {initial if agent_color == game.board().turn else -initial:+d} cp")
    for drop, ply, played_move, bestmove, score in sorted(drops, reverse=True)[:10]:
        print(
            f"ply {ply:3} played {played_move}, teacher {bestmove}: "
            f"drop {drop:+5d} cp, resulting score {score:+5d} cp"
        )


if __name__ == "__main__":
    main()
