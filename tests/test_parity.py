"""Exact parity between the numba NNUE kernels and the numpy reference.

These are the guarantees the evaluation rests on:
1. The jitted forward pass equals reference.forward_int bit for bit.
2. Incremental accumulator updates along a game equal a from-scratch rebuild, so the
   search can trust update_ply at any depth.
"""

import random
import time

import chess
import numpy as np
import pytest

import bitboard as bb
import nnue_bitboard as nb
from nnue_net import NetworkWeights
from tests.reference import forward_int
from tools.gen_random_net import random_network

_NET = random_network(2026)


class Tracker:
    """A board stack plus accumulators driven move by move, the way the kernel does it."""

    def __init__(self, net: NetworkWeights) -> None:
        self.net = net
        self.board = bb.new_stacks()
        self.acc = nb.new_acc_stacks()
        self.moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
        self.ply = 0

    def set_position(self, board: chess.Board) -> None:
        self.ply = 0
        bb.set_from_board(board, self.board, 0)
        nb.refresh_ply(
            self.net.ft_w,
            self.net.ft_b,
            self.net.psqt_w,
            self.board.mailbox,
            self.board.state,
            0,
            self.acc.white_acc,
            self.acc.black_acc,
            self.acc.white_psqt,
            self.acc.black_psqt,
        )

    def push(self, move: chess.Move) -> None:
        s = self.board
        count = bb.generate_moves(
            s.pieces, s.occupied, s.mailbox, s.state, self.ply, self.moves, False
        )
        ours = [
            int(m) for m in self.moves[self.ply, :count] if bb.move_to_uci(int(m)) == move.uci()
        ]
        assert len(ours) == 1, move.uci()
        assert bb.make_move(s.pieces, s.occupied, s.mailbox, s.state, s.keys, self.ply, ours[0])
        nb.update_ply(
            self.net.ft_w,
            self.net.ft_b,
            self.net.psqt_w,
            s.mailbox,
            s.state,
            self.ply,
            ours[0],
            self.acc.white_acc,
            self.acc.black_acc,
            self.acc.white_psqt,
            self.acc.black_psqt,
        )
        self.ply += 1

    def pop(self) -> None:
        self.ply -= 1

    def evaluate(self) -> int:
        net = self.net
        return int(
            nb.evaluate(
                self.ply,
                self.board.state,
                self.board.occupied,
                self.acc.white_acc,
                self.acc.black_acc,
                self.acc.white_psqt,
                self.acc.black_psqt,
                self.acc.act,
                self.acc.l1c,
                self.acc.l1x,
                self.acc.l2c,
                self.acc.l2x,
                net.l1_w,
                net.l1_b,
                net.l2_w,
                net.l2_b,
                net.out_w,
                net.out_b,
            )
        )


@pytest.fixture(scope="module")
def tracker() -> Tracker:
    return Tracker(_NET)


def test_forward_matches_reference_on_fixed_positions(tracker: Tracker) -> None:
    fens = [
        chess.STARTING_FEN,
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "8/2k5/8/8/8/8/2K5/1Q6 b - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        tracker.set_position(board)
        assert tracker.evaluate() == forward_int(_NET, board), fen
        assert nb.evaluate_board(_NET, board) == forward_int(_NET, board), fen


def test_forward_matches_reference_on_random_positions(tracker: Tracker) -> None:
    rng = random.Random(11)
    checked = 0
    for _game in range(4):
        board = chess.Board()
        for _ply in range(100):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if rng.random() < 0.2:
                tracker.set_position(board)
                assert tracker.evaluate() == forward_int(_NET, board), board.fen()
                checked += 1
    assert checked > 30


def test_incremental_equals_refresh_along_games(tracker: Tracker) -> None:
    rng = random.Random(23)
    for _game in range(6):
        board = chess.Board()
        tracker.set_position(board)
        for _ply in range(110):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            tracker.push(move)
            board.push(move)
            assert tracker.evaluate() == forward_int(_NET, board), (
                f"{board.fen()} after {move.uci()}"
            )


def test_special_moves_incremental(tracker: Tracker) -> None:
    scenarios = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8", "a1d1", "d8d1"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6", "e8d7"]),
        # The king stands on e2 so the rook can reach b1 along the first rank.
        ("4kn2/6P1/8/8/8/8/1p2K3/7R b - - 0 1", ["b2b1q", "h1b1", "f8e6"]),
    ]
    for fen, ucis in scenarios:
        board = chess.Board(fen)
        tracker.set_position(board)
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            tracker.push(move)
            board.push(move)
            assert tracker.evaluate() == forward_int(_NET, board), f"{fen} {uci}"


def test_push_pop_restores_score(tracker: Tracker) -> None:
    board = chess.Board()
    tracker.set_position(board)
    before = tracker.evaluate()
    for move in list(board.legal_moves):
        tracker.push(move)
        tracker.evaluate()
        tracker.pop()
    assert tracker.evaluate() == before


def test_speed_smoke(tracker: Tracker) -> None:
    board = chess.Board()
    tracker.set_position(board)
    count = 20_000
    start = time.perf_counter()
    for _ in range(count):
        tracker.evaluate()
    eval_rate = count / (time.perf_counter() - start)

    moves = list(board.legal_moves)
    cycles = 2_000
    start = time.perf_counter()
    for i in range(cycles):
        tracker.push(moves[i % len(moves)])
        tracker.evaluate()
        tracker.pop()
    node_rate = cycles / (time.perf_counter() - start)

    print(f"\nevaluate only: {eval_rate:,.0f}/s | push+eval+pop: {node_rate:,.0f}/s")
    assert eval_rate > 5_000
    assert node_rate > 1_000
