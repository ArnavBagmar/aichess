"""Play reversed-color pairs from curated FENs with correctly attributed failures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.sandbox import local
from tools.benchmark import POSITIONS, Position


def agent_score(outcome: Outcome, agent_is_white: bool) -> float:
    if outcome.result in {"draw", "void"}:
        return 0.5
    return float((outcome.result == "white") == agent_is_white)


def failed_side(outcome: Outcome) -> str | None:
    if outcome.termination not in FAILED_TERMINATIONS:
        return None
    if outcome.result == "white":
        return "black"
    if outcome.result == "black":
        return "white"
    return "both"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--base-ms", type=int, default=5_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--pgn-dir", type=Path)
    parser.add_argument(
        "--fen-jsonl", type=Path, help="optional JSONL records containing a fen field"
    )
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    if arguments.fen_jsonl:
        records = (
            json.loads(line)
            for line in arguments.fen_jsonl.read_text(encoding="utf-8").splitlines()
        )
        selected = tuple(
            Position(f"dataset_{index:04}", "dataset", record["fen"])
            for index, record in enumerate(records, 1)
        )[: arguments.positions]
    else:
        selected = POSITIONS[: max(1, min(arguments.positions, len(POSITIONS)))]
    if not selected:
        raise SystemExit("no starting positions selected")
    pentanomial = [0, 0, 0, 0, 0]
    game_scores: list[float] = []
    agent_failures: dict[str, int] = {}
    opponent_failures: dict[str, int] = {}

    if arguments.pgn_dir:
        arguments.pgn_dir.mkdir(parents=True, exist_ok=True)

    for pair_index, position in enumerate(selected, 1):
        pair_score = 0.0
        for game_in_pair, agent_is_white in enumerate((True, False), 1):
            white = agent if agent_is_white else opponent
            black = opponent if agent_is_white else agent
            outcome = play_match(
                local(white),
                local(black),
                arguments.base_ms,
                arguments.increment_ms,
                start_fen=position.fen,
            )
            score = agent_score(outcome, agent_is_white)
            game_scores.append(score)
            pair_score += score
            failure = failed_side(outcome)
            if failure is not None:
                failed_agent = failure == "both" or (failure == "white") == agent_is_white
                destination = agent_failures if failed_agent else opponent_failures
                destination[outcome.termination] = destination.get(outcome.termination, 0) + 1
            print(
                f"pair {pair_index}/{len(selected)} game {game_in_pair}: {position.name}, "
                f"agent {'white' if agent_is_white else 'black'}, score {score:g}, "
                f"{outcome.termination}"
            )
            if arguments.pgn_dir:
                filename = f"{pair_index:02}-{position.name}-{game_in_pair}.pgn"
                (arguments.pgn_dir / filename).write_text(outcome.pgn + "\n", encoding="utf-8")
        pentanomial[round(pair_score * 2)] += 1

    score = statistics.fmean(game_scores)
    wins = sum(value == 1 for value in game_scores)
    draws = sum(value == 0.5 for value in game_scores)
    losses = sum(value == 0 for value in game_scores)
    if 0 < score < 1:
        elo = -400 * math.log10(1 / score - 1)
        elo_text = f"{elo:+.1f} unanchored Elo"
    else:
        elo_text = "Elo undefined at a 0% or 100% sample score"
    print(f"\n+{wins} ={draws} -{losses}, score {score:.1%}, {elo_text}")
    print(f"pentanomial [LL, LD, DD/WL, WD, WW]: {pentanomial}")
    print(f"agent failures: {agent_failures or 'none'}")
    print(f"opponent failures: {opponent_failures or 'none'}")


if __name__ == "__main__":
    main()
