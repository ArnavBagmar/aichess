"""The submission entrypoint. The platform imports this file and calls get_move.

Phase 1: NNUE evaluation (nnue_engine + weights/nnue.npz) behind a one-ply move choice.
The weights are still randomly initialized, so strength is not the point yet; this wires
the full pipeline (weights -> incremental accumulators -> jitted forward) into the game
loop and proves it inside the platform's constraints. Phase 3 replaces the one-ply loop
with a real search and starts using time_left_ms.
"""

import random

import chess

from nnue_engine import load_engine, warm_up

MATE_SCORE = 1_000_000

# Import time runs once per game, inside a 60 second budget, before the clock starts.
# Loading weights and compiling the numba kernels both happen here (~1s locally).
_engine = load_engine()
warm_up(_engine)


def _nnue_move(fen: str) -> str:
    _engine.set_position(fen)
    moves = list(_engine.board.legal_moves)
    if not moves:
        raise ValueError(f"no legal moves in {fen!r}")
    best_score = -2 * MATE_SCORE
    best: list[chess.Move] = []
    for move in moves:
        _engine.push(move)
        board = _engine.board
        if board.is_checkmate():
            score = MATE_SCORE
        elif (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            # Dead until move history survives across calls (Phase 3): a board built
            # from a bare FEN cannot see repetitions. Kept so the scoring stays right
            # once that state exists.
            or board.is_fivefold_repetition()
        ):
            score = 0
        else:
            score = -_engine.evaluate()  # stm-relative, so negate after the push
        _engine.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
    return random.choice(best).uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation, chosen by one-ply NNUE evaluation."""
    try:
        return _nnue_move(fen)
    except Exception as error:
        # A bug in our pipeline should cost strength, not the game. stdout is safe:
        # the runner redirects it away from the protocol stream.
        print(f"nnue move selection failed ({error!r}); playing a random legal move")
        return random.choice(list(chess.Board(fen).legal_moves)).uci()
