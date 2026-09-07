"""Rapidly screen one engine against held-out MultiPV teacher positions."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import islice
from pathlib import Path

import chess

from agent import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--skip-positions", type=int, default=0)
    parser.add_argument("--clock-ms", type=int, default=1_000)
    parser.add_argument(
        "--min-gap-cp",
        type=int,
        default=0,
        help="require this centipawn gap between the teacher's first and second moves",
    )
    parser.add_argument("--misses-output", type=Path)
    args = parser.parse_args()

    records = (json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines())
    eligible = (
        record
        for record in records
        if len(record.get("candidates", ())) >= 2
        and int(record["candidates"][0]["score_cp"]) - int(record["candidates"][1]["score_cp"])
        >= args.min_gap_cp
    )
    selected = list(islice(eligible, args.skip_positions, args.skip_positions + args.positions))
    if not selected:
        raise SystemExit("no positions selected")

    top1 = 0
    top3 = 0
    known_regrets: list[int] = []
    reference_top1 = 0
    reference_regrets: list[int] = []
    depths: list[int] = []
    nodes: list[int] = []
    misses: list[dict[str, object]] = []
    for record in selected:
        board = chess.Board(record["fen"])
        engine = Engine()
        move = engine.choose(board, args.clock_ms).uci()
        candidates = {item["move"]: item for item in record.get("candidates", ())}
        source = record.get("source")
        reference_move = source.get("played_move") if isinstance(source, dict) else None
        if isinstance(reference_move, str):
            reference_top1 += int(reference_move == record["bestmove"])
            reference = candidates.get(reference_move)
            if reference is not None:
                reference_regrets.append(int(record["score_cp"]) - int(reference["score_cp"]))
        if move == record["bestmove"]:
            top1 += 1
        else:
            misses.append(
                {
                    "fen": record["fen"],
                    "selected_move": move,
                    "best_move": record["bestmove"],
                    "best_score_cp": record["score_cp"],
                    "selected_score_cp": (
                        candidates[move]["score_cp"] if move in candidates else None
                    ),
                    "reference_move": reference_move,
                    "best_gap_cp": int(record["candidates"][0]["score_cp"])
                    - int(record["candidates"][1]["score_cp"]),
                    "completed_depth": engine.stats.completed_depth,
                }
            )
        candidate = candidates.get(move)
        if candidate is not None:
            top3 += 1
            known_regrets.append(int(record["score_cp"]) - int(candidate["score_cp"]))
        depths.append(engine.stats.completed_depth)
        nodes.append(engine.stats.nodes)

    count = len(selected)
    print(
        f"positions={count} min_gap_cp={args.min_gap_cp} "
        f"top1={top1 / count:.1%} top3={top3 / count:.1%} "
        f"known_regret_cp={statistics.fmean(known_regrets) if known_regrets else float('nan'):.1f} "
        f"reference_top1={reference_top1 / count:.1%} "
        f"reference_regret_cp="
        f"{statistics.fmean(reference_regrets) if reference_regrets else float('nan'):.1f} "
        f"mean_depth={statistics.fmean(depths):.2f} mean_nodes={statistics.fmean(nodes):.0f}"
    )
    if args.misses_output:
        args.misses_output.parent.mkdir(parents=True, exist_ok=True)
        args.misses_output.write_text(json.dumps(misses, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
