"""Estimate the agent's rating by playing it against Stockfish under tournament rules.

Stockfish is a sparring partner and never ships. `harness/package.py` packages
root-level `*.py` plus `weights/`, so nothing under `tools/` can reach submission.zip,
and the binary itself lives outside the repository. The competition ban is on engines
that run inside the upload; measuring against one locally is not that.

Games run through `harness.referee.play_match`, so the clock, flag-fall, illegal-move
loss and 300-ply material adjudication are the platform's own rules rather than a
re-implementation. Stockfish is wrapped as a `harness.sandbox.Agent` so the referee
cannot tell the two sides apart.

Our search is deterministic, so every game from the standard start would be the same
game. Openings are therefore seeded random positions filtered to be near-equal by our
own evaluation, and each is played twice with colours reversed, which is both the
standard way to cut variance and closer to the event's curated start positions.

Usage:
    uv run python tools/elo_bench.py --stockfish PATH --elo 1600 --games 20
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness.referee import Outcome, play_match  # noqa: E402
from harness.rules import BASE_MS, INCREMENT_MS, PLY_CAP  # noqa: E402
from harness.sandbox import Agent, AgentFailure, local  # noqa: E402

# Positions are accepted as openings only if our own eval calls them near-equal, so a
# pair is not decided before it starts. In centipawns.
BALANCE_CP = 90.0
OPENING_PLIES = 4

FAILURE_TERMINATIONS = frozenset({"crash", "illegal", "flag", "init", "both_failed"})


class StockfishAgent(Agent):
    """Stockfish behind the harness's Agent interface, so the referee stays untouched.

    The referee hands each side only its own remaining clock, which is what Stockfish's
    time manager needs for its own move; the opponent's clock is mirrored from it.
    """

    def __init__(self, path: Path, elo: int, increment_ms: int) -> None:
        super().__init__([str(path)])
        self.path = path
        self.elo = elo
        self.increment_ms = increment_ms
        self._engine: chess.engine.SimpleEngine | None = None

    def start(self, init_budget_s: float) -> None:
        engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        # One core and a small table, matching the constraints our own agent runs under.
        engine.configure({"Threads": 1, "Hash": 16})
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        self._engine = engine

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._engine is None:
            raise RuntimeError("stockfish moved before start")
        board = chess.Board(fen)
        clock = time_left_ms / 1000.0
        increment = self.increment_ms / 1000.0
        limit = chess.engine.Limit(
            white_clock=clock, black_clock=clock, white_inc=increment, black_inc=increment
        )
        try:
            played = self._engine.play(board, limit)
        except chess.engine.EngineError as error:
            raise AgentFailure(f"stockfish: {error}") from error
        if played.move is None:
            raise AgentFailure("stockfish returned no move")
        return played.move.uci()

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None


@dataclass(frozen=True)
class Tally:
    """Results from our agent's point of view."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    failures: tuple[str, ...] = ()

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0


def elo_difference(score: float) -> float:
    """Rating difference implied by a score, in Elo. Infinite at a clean sweep."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_margin(tally: Tally) -> float:
    """One standard error on the score, from the spread of per-game results."""
    if tally.games < 2:
        return 0.0
    mean = tally.score
    outcomes = [1.0] * tally.wins + [0.5] * tally.draws + [0.0] * tally.losses
    variance = sum((value - mean) ** 2 for value in outcomes) / (tally.games - 1)
    return math.sqrt(variance / tally.games)


class Balancer:
    """Our own evaluation, used only to keep opening positions near-equal."""

    def __init__(self) -> None:
        from nnue_engine import load_engine, warm_up

        self._engine = load_engine()
        warm_up(self._engine)

    def __call__(self, fen: str) -> float:
        self._engine.set_position(fen)
        return self._engine.evaluate_cp()


def opening_positions(count: int, seed: int, evaluate: Balancer) -> list[str]:
    """Seeded near-equal positions, so a deterministic agent plays varied games."""
    rng = random.Random(seed)
    positions: list[str] = []
    attempts = 0
    while len(positions) < count and attempts < count * 200:
        attempts += 1
        board = chess.Board()
        for _ in range(OPENING_PLIES):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        fen = board.fen()
        if abs(evaluate(fen)) <= BALANCE_CP and fen not in positions:
            positions.append(fen)
    if len(positions) < count:
        raise RuntimeError(f"only found {len(positions)} balanced openings of {count} wanted")
    return positions


def play_pair(
    agent_dir: Path,
    stockfish: Path,
    elo: int,
    fen: str,
    base_ms: int,
    increment_ms: int,
) -> list[tuple[Outcome, bool]]:
    """The same opening twice, our agent as white then as black."""
    played: list[tuple[Outcome, bool]] = []
    for we_are_white in (True, False):
        us = local(agent_dir)
        them = StockfishAgent(stockfish, elo, increment_ms)
        white, black = (us, them) if we_are_white else (them, us)
        outcome = play_match(white, black, base_ms, increment_ms, PLY_CAP, start_fen=fen)
        played.append((outcome, we_are_white))
    return played


def record(tally: Tally, outcome: Outcome, we_were_white: bool) -> Tally:
    ours = "white" if we_were_white else "black"
    theirs = "black" if we_were_white else "white"
    failures = tally.failures
    # A crash, illegal move or flag loses the same game on the platform, so it counts as
    # a loss and is surfaced rather than quietly folded into the score.
    if outcome.termination in FAILURE_TERMINATIONS:
        failures = (*failures, outcome.termination)
    if outcome.result == ours:
        return Tally(tally.wins + 1, tally.draws, tally.losses, failures)
    if outcome.result == theirs:
        return Tally(tally.wins, tally.draws, tally.losses + 1, failures)
    return Tally(tally.wins, tally.draws + 1, tally.losses, failures)


def report(tally: Tally, elo: int) -> None:
    margin = score_margin(tally)
    centre = elo_difference(tally.score)
    low = elo_difference(max(0.0, tally.score - margin))
    high = elo_difference(min(1.0, tally.score + margin))

    print(f"\n{tally.games} games: +{tally.wins} ={tally.draws} -{tally.losses}")
    print(f"score {tally.score:.1%}")
    if math.isinf(centre):
        bound = "above" if centre > 0 else "below"
        print(f"estimated rating: {bound} Stockfish {elo} (the result does not bracket it)")
    else:
        print(f"rating difference: {centre:+.0f} Elo (1 SE: {low:+.0f} to {high:+.0f})")
        print(
            f"estimated rating: {elo + centre:.0f} "
            f"({elo + low:.0f} to {elo + high:.0f})"
        )
    if tally.failures:
        print(f"\nWARNING: {len(tally.failures)} games lost to agent failure: {tally.failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--elo", type=int, required=True, help="Stockfish UCI_Elo to face")
    parser.add_argument("--games", type=int, default=20, help="rounded up to a colour pair")
    parser.add_argument("--base-ms", type=int, default=BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=INCREMENT_MS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pgn", type=Path, default=None)
    arguments = parser.parse_args()

    pairs = max(1, (arguments.games + 1) // 2)
    openings = opening_positions(pairs, arguments.seed, Balancer())

    print(
        f"agent vs Stockfish UCI_Elo {arguments.elo} | {pairs * 2} games "
        f"| {arguments.base_ms / 1000:.0f}s + {arguments.increment_ms / 1000:.1f}s",
        flush=True,
    )

    tally = Tally()
    games: list[str] = []
    for index, fen in enumerate(openings, start=1):
        for outcome, we_were_white in play_pair(
            arguments.agent,
            arguments.stockfish,
            arguments.elo,
            fen,
            arguments.base_ms,
            arguments.increment_ms,
        ):
            tally = record(tally, outcome, we_were_white)
            games.append(outcome.pgn)
        print(
            f"pair {index}/{pairs}: +{tally.wins} ={tally.draws} -{tally.losses} "
            f"({tally.score:.1%})",
            flush=True,
        )

    if arguments.pgn is not None:
        arguments.pgn.write_text("\n\n".join(games), encoding="utf-8")

    report(tally, arguments.elo)


if __name__ == "__main__":
    main()
