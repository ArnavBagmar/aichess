"""Exact parity between the numba engine and the numpy reference.

These are the guarantees Phase 1 exists to establish:
1. The jitted forward pass equals reference.forward_int bit for bit.
2. Incremental accumulator updates along a game equal a from-scratch rebuild, so a
   search can trust push/pop at any depth.
"""

import random
import time

import chess
import pytest

from nnue_engine import Engine, warm_up
from tests.reference import forward_int
from tools.gen_random_net import random_network

_NET = random_network(2026)


def _fresh_engine() -> Engine:
    engine = Engine(_NET)
    warm_up(engine)
    return engine


@pytest.fixture(scope="module")
def engine() -> Engine:
    return _fresh_engine()


def test_forward_matches_reference_on_fixed_positions(engine: Engine) -> None:
    fens = [
        chess.STARTING_FEN,
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "8/2k5/8/8/8/8/2K5/1Q6 b - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    ]
    for fen in fens:
        engine.set_position(fen)
        assert engine.evaluate() == forward_int(_NET, chess.Board(fen)), fen


def test_forward_matches_reference_on_random_positions(engine: Engine) -> None:
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
                engine.set_position(board.fen())
                assert engine.evaluate() == forward_int(_NET, board), board.fen()
                checked += 1
    assert checked > 30


def test_incremental_equals_refresh_along_games(engine: Engine) -> None:
    rng = random.Random(23)
    rebuild = _fresh_engine()
    for _game in range(6):
        engine.set_position(chess.STARTING_FEN)
        board = chess.Board()
        for _ply in range(110):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            engine.push(move)
            board.push(move)
            incremental = engine.evaluate()
            rebuild.set_position(board.fen())
            assert incremental == rebuild.evaluate(), f"{board.fen()} after {move.uci()}"


def test_special_moves_incremental(engine: Engine) -> None:
    scenarios = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8", "a1d1", "d8d1"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6", "e8d7"]),
        ("4kn2/6P1/8/8/8/8/1p6/4K2R b K - 0 1", ["b2b1q", "h1b1", "f8e6"]),
    ]
    rebuild = _fresh_engine()
    for fen, ucis in scenarios:
        engine.set_position(fen)
        board = chess.Board(fen)
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            engine.push(move)
            board.push(move)
            rebuild.set_position(board.fen())
            assert engine.evaluate() == rebuild.evaluate(), f"{fen} {uci}"


def test_push_pop_restores_score(engine: Engine) -> None:
    engine.set_position(chess.STARTING_FEN)
    before = engine.evaluate()
    for move in list(engine.board.legal_moves):
        engine.push(move)
        engine.evaluate()
        engine.pop()
    assert engine.evaluate() == before


def test_speed_smoke(engine: Engine) -> None:
    engine.set_position(chess.STARTING_FEN)
    count = 20_000
    start = time.perf_counter()
    for _ in range(count):
        engine.evaluate()
    eval_rate = count / (time.perf_counter() - start)

    moves = list(engine.board.legal_moves)
    cycles = 2_000
    start = time.perf_counter()
    for i in range(cycles):
        engine.push(moves[i % len(moves)])
        engine.evaluate()
        engine.pop()
    node_rate = cycles / (time.perf_counter() - start)

    print(f"\nevaluate only: {eval_rate:,.0f}/s | push+eval+pop: {node_rate:,.0f}/s")
    assert eval_rate > 5_000
    assert node_rate > 1_000
