"""The submission entrypoint. The platform imports this file and calls get_move.

Move choice is an alpha-beta search over the trained NNUE evaluation. The board, move
generation, accumulator updates and the search loop itself are numba kernels over
bitboards (`bitboard.py`, `nnue_bitboard.py`, `search_kernel.py`); `search.py` is the
thin Python root that runs iterative deepening and aspiration windows within the clock
and checks the kernel's move against python-chess before returning it. The searcher
owns the transposition table and the record of positions we have been asked about,
which is what makes repetitions visible; it lives for exactly one game.
"""

import random

import chess

from search import load_searcher

# Import time runs once per game, inside a 60 second budget, before the clock starts.
# Loading weights, compiling the numba kernels, and warming the search all happen here.
_searcher = load_searcher()
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
