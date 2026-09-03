"""Cross-validation of the integer pipeline against the trainer's float forward.

If the integer derivation in nnue_arch/reference is structurally wrong (a shift off by
one, a sign flipped, a concat out of order) the two references diverge by hundreds of
units on nearly every position. The trainer's 1e-5-before-floor epsilon makes rare
one-grid-step differences legitimate, so the assertions bound the disagreement instead
of demanding exact equality.
"""

import random

import chess
import pytest

from tests.reference import forward_float, forward_int
from tools.gen_random_net import random_network

_FIXED_FENS = [
    chess.STARTING_FEN,
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",  # 2 pieces, bucket 0
    "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",  # 3 pieces, black to move
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "8/2k5/8/8/8/8/2K5/1Q6 b - - 0 1",
]


def _sampled_positions(games: int, plies: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = [chess.Board(fen) for fen in _FIXED_FENS]
    for _ in range(games):
        board = chess.Board()
        for _ply in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if rng.random() < 0.25:
                boards.append(board.copy(stack=False))
    return boards


@pytest.mark.parametrize("net_seed", [1, 2, 3])
def test_int_forward_matches_trainer_float(net_seed: int) -> None:
    net = random_network(net_seed)
    boards = _sampled_positions(games=4, plies=90, seed=net_seed + 100)
    assert len(boards) > 40

    diffs = []
    for board in boards:
        score_int = forward_int(net, board)
        score_float = forward_float(net, board)
        diffs.append(abs(score_int - score_float))

    close = sum(1 for d in diffs if d <= 2.0)
    # One unit is 1/32 cp; anything structural would blow past these bounds at once.
    assert max(diffs) <= 200.0, f"max diff {max(diffs)}"
    assert close / len(diffs) >= 0.99, f"only {close}/{len(diffs)} within 2 units"
