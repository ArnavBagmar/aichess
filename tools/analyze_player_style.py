"""Summarize a player's move and clock style from public PGNs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from tools.analyze_pgns import phase


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_dir", type=Path)
    parser.add_argument("--player", required=True)
    parser.add_argument("--initial-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    moves = Counter[str]()
    pieces = Counter[str]()
    phases = Counter[str]()
    phase_styles: dict[str, Counter[str]] = {
        name: Counter() for name in ("opening", "middlegame", "endgame")
    }
    terminations = Counter[str]()
    spends: list[float] = []
    spend_by_clock: dict[str, list[float]] = {"over_60s": [], "10_to_60s": [], "under_10s": []}
    games = wins = draws = losses = 0
    lengths: list[int] = []
    for path in sorted(args.pgn_dir.glob("*.pgn")):
        with path.open(encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            continue
        white = game.headers.get("White") == args.player
        black = game.headers.get("Black") == args.player
        if not white and not black:
            continue
        color = chess.WHITE if white else chess.BLACK
        result = game.headers.get("Result")
        games += 1
        wins += int(result == ("1-0" if white else "0-1"))
        losses += int(result == ("0-1" if white else "1-0"))
        draws += int(result == "1/2-1/2")
        terminations[game.headers.get("Termination", "unknown")] += 1
        lengths.append(game.end().board().ply() - game.board().ply())
        previous_clock = args.initial_ms / 1000.0
        board = game.board()
        for node in game.mainline():
            move = node.move
            if board.turn == color:
                move_phase = phase(board)
                capture = int(board.is_capture(move))
                check = int(board.gives_check(move))
                moves["total"] += 1
                moves["capture"] += capture
                moves["check"] += check
                moves["promotion"] += int(move.promotion is not None)
                moves["castle"] += int(board.is_castling(move))
                moves["quiet"] += int(not board.is_capture(move) and move.promotion is None)
                piece = board.piece_type_at(move.from_square) or 0
                pieces[chess.piece_name(piece)] += 1
                phases[move_phase] += 1
                phase_styles[move_phase]["capture"] += capture
                phase_styles[move_phase]["check"] += check
                current_clock = node.clock()
                if current_clock is not None:
                    spent = max(0.0, previous_clock + args.increment_ms / 1000.0 - current_clock)
                    spends.append(spent)
                    bucket = (
                        "over_60s"
                        if previous_clock > 60
                        else "10_to_60s"
                        if previous_clock > 10
                        else "under_10s"
                    )
                    spend_by_clock[bucket].append(spent)
                    previous_clock = current_clock
            board.push(move)

    total = max(1, moves["total"])
    result_payload = {
        "player": args.player,
        "games": games,
        "record": {"wins": wins, "draws": draws, "losses": losses},
        "score_percent": round(100 * (wins + draws / 2) / max(1, games), 1),
        "mean_game_plies": round(statistics.fmean(lengths), 1),
        "terminations": dict(terminations),
        "moves": {
            "total": moves["total"],
            **{
                name: {"count": moves[name], "percent": round(100 * moves[name] / total, 1)}
                for name in ("quiet", "capture", "check", "promotion", "castle")
            },
        },
        "piece_move_percent": {
            name: round(100 * count / total, 1) for name, count in pieces.most_common()
        },
        "phase_moves": dict(phases),
        "phase_style_percent": {
            name: {
                category: round(100 * counts[category] / max(1, phases[name]), 1)
                for category in ("capture", "check")
            }
            for name, counts in phase_styles.items()
        },
        "clock_spend_seconds": {
            "median": round(statistics.median(spends), 3),
            "p90": round(percentile(spends, 0.9), 3),
            "maximum": round(max(spends), 3),
            "mean_by_remaining_clock": {
                name: round(statistics.fmean(values), 3) if values else None
                for name, values in spend_by_clock.items()
            },
        },
    }
    rendered = json.dumps(result_payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
