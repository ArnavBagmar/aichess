"""Nodes per second on four fixed positions, for before-and-after speed comparisons.

Not a strength measure: use the SPRT for that. This only says whether a change made
the search faster or slower per node, which the SPRT cannot separate from smarter.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import search  # noqa: E402
from nnue_engine import load_engine  # noqa: E402

POSITIONS = {
    "opening": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "middlegame": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
    # Sharp, but with no forced mate in reach: a mate ends the search early and the
    # probe would time a handful of nodes.
    "tactical": "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "endgame": "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
}
CLOCK_MS = 60_000  # budget_ms(60_000) is about 2.25 s per position


def main() -> None:
    searcher = search.Searcher(load_engine())
    searcher.warm_up()
    total_nodes = 0
    total_seconds = 0.0
    for name, fen in POSITIONS.items():
        started = time.perf_counter()
        searcher.pick(fen, CLOCK_MS)
        elapsed = time.perf_counter() - started
        total_nodes += searcher.nodes
        total_seconds += elapsed
        print(f"{name:<11} {searcher.nodes:>8} nodes  {searcher.nodes / elapsed / 1000:6.1f} knps")
    print(f"{'overall':<11} {total_nodes:>8} nodes  {total_nodes / total_seconds / 1000:6.1f} knps")


if __name__ == "__main__":
    main()
