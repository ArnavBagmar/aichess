"""Measure candidate move loss in paired-arena PGNs with an offline UCI engine."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn

PHASE_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def phase(board: chess.Board) -> str:
    remaining = sum(
        weight * len(board.pieces(piece_type, color))
        for piece_type, weight in PHASE_WEIGHT.items()
        for color in chess.COLORS
    )
    if board.fullmove_number <= 15 and remaining >= 18:
        return "opening"
    if remaining <= 8:
        return "endgame"
    return "middlegame"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("pgn_dirs", nargs="+", type=Path)
    parser.add_argument("--time-ms", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--player",
        help="analyze only this named player's moves instead of paired-arena filename sides",
    )
    args = parser.parse_args()

    totals: dict[str, list[int]] = defaultdict(list)
    records: list[dict[str, int | str]] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        for directory in args.pgn_dirs:
            for path in sorted(directory.glob("*.pgn")):
                with path.open(encoding="utf-8") as handle:
                    game = chess.pgn.read_game(handle)
                if game is None:
                    continue
                if args.player:
                    if game.headers.get("White") == args.player:
                        agent_color = chess.WHITE
                    elif game.headers.get("Black") == args.player:
                        agent_color = chess.BLACK
                    else:
                        continue
                else:
                    agent_color = chess.WHITE if path.stem.endswith("-1") else chess.BLACK
                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == agent_color:
                        before = engine.analyse(
                            board, chess.engine.Limit(time=args.time_ms / 1000.0)
                        )["score"].pov(agent_color).score(mate_score=100_000)
                        move_phase = phase(board)
                        san = board.san(move)
                        board.push(move)
                        after = engine.analyse(
                            board, chess.engine.Limit(time=args.time_ms / 1000.0)
                        )["score"].pov(agent_color).score(mate_score=100_000)
                        if before is not None and after is not None:
                            loss = max(0, before - after)
                            totals[move_phase].append(loss)
                            records.append(
                                {
                                    "game": path.name,
                                    "ply": board.ply(),
                                    "phase": move_phase,
                                    "move": san,
                                    "before_cp": before,
                                    "after_cp": after,
                                    "loss_cp": loss,
                                }
                            )
                    else:
                        board.push(move)
    finally:
        engine.quit()

    summary: dict[str, object] = {}
    for name, losses in sorted(totals.items()):
        ordered = sorted(losses)
        competitive = [
            int(record["loss_cp"])
            for record in records
            if record["phase"] == name
            and int(record["before_cp"]) > -500
            and abs(int(record["before_cp"])) < 90_000
            and abs(int(record["after_cp"])) < 90_000
        ]
        summary[name] = {
            "moves": len(losses),
            "mean_loss_cp": round(sum(losses) / len(losses), 2),
            "median_loss_cp": ordered[len(ordered) // 2],
            "mistakes_100cp": sum(loss >= 100 for loss in losses),
            "blunders_200cp": sum(loss >= 200 for loss in losses),
            "competitive_moves": len(competitive),
            "competitive_mean_loss_cp": round(sum(competitive) / len(competitive), 2),
            "competitive_mistakes_100cp": sum(loss >= 100 for loss in competitive),
            "mate_score_swings": sum(loss >= 90_000 for loss in losses),
        }
    largest = sorted(records, key=lambda x: int(x["loss_cp"]), reverse=True)[:50]
    result = {"summary": summary, "largest_losses": largest}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
