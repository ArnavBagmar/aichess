"""The submission entrypoint. The platform imports this file and calls get_move.

Move choice is an alpha-beta search (`search.py`) over the trained NNUE evaluation
(`nnue_engine` + `weights/nnue.npz`): iterative deepening to whatever depth the clock
allows, with a quiescence search at the leaves so positions are never scored in the
middle of an exchange. The searcher owns the transposition table and the record of
positions we have been asked about, which is what makes repetitions visible; it lives
for exactly one game.
"""

import random

import chess

from nnue_engine import load_engine, warm_up
from search import Searcher

# Import time runs once per game, inside a 60 second budget, before the clock starts.
# Loading weights, compiling the numba kernels, and warming the search all happen here.
_engine = load_engine()
warm_up(_engine)
_searcher = Searcher(_engine)
_searcher.warm_up()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation, chosen by alpha-beta search."""
    try:
        return _searcher.pick(fen, time_left_ms).uci()
    except Exception as error:
        # A bug in our search should cost strength, not the game. stdout is safe:
        # the runner redirects it away from the protocol stream.
        print(f"search failed ({error!r}); playing a random legal move")
        return random.choice(list(chess.Board(fen).legal_moves)).uci()
