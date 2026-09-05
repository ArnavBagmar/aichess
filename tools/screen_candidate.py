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
    args = parser.parse_args()

    records = (
        json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
    )
    eligible = (
        record
        for record in records
        if len(record.get("candidates", ())) >= 2
        and int(record["candidates"][0]["score_cp"])
        - int(record["candidates"][1]["score_cp"])
        >= args.min_gap_cp
    )
    selected = list(islice(eligible, args.skip_positions, args.skip_positions + args.positions))
    if not selected:
        raise SystemExit("no positions selected")

    top1 = 0
    top3 = 0
    known_regrets: list[int] = []
    depths: list[int] = []
    nodes: list[int] = []
    for record in selected:
        board = chess.Board(record["fen"])
        engine = Engine()
        move = engine.choose(board, args.clock_ms).uci()
        candidates = {item["move"]: item for item in record.get("candidates", ())}
        if move == record["bestmove"]:
            top1 += 1
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
        f"mean_depth={statistics.fmean(depths):.2f} mean_nodes={statistics.fmean(nodes):.0f}"
    )


if __name__ == "__main__":
    main()
