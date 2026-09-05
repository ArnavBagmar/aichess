"""The bitboard NNUE must equal the python-chess engine exactly, position by position."""

import random

import chess
import numpy as np

import bitboard as bb
import nnue_bitboard as nb
from nnue_engine import Engine
from nnue_features import active_features
from tools.gen_random_net import random_network

_NET = random_network(2026)


def test_feature_index_matches_nnue_features() -> None:
    rng = random.Random(5)
    board = chess.Board()
    for _ in range(200):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        for pov in (True, False):
            king = board.king(chess.WHITE if pov else chess.BLACK)
            assert king is not None
            expected = sorted(active_features(board, pov))
            got = sorted(
                nb.feature_index(0 if pov else 1, king, sq, bb.piece_code(piece))
                for sq, piece in board.piece_map().items()
            )
            assert got == expected, board.fen()


def evaluate(stacks: bb.BoardStacks, acc: nb.AccStacks, ply: int) -> int:
    return int(
        nb.evaluate(
            ply,
            stacks.state,
            stacks.occupied,
            acc.white_acc,
            acc.black_acc,
            acc.white_psqt,
            acc.black_psqt,
            acc.act,
            acc.l1c,
            acc.l1x,
            acc.l2c,
            acc.l2x,
            _NET.l1_w,
            _NET.l1_b,
            _NET.l2_w,
            _NET.l2_b,
            _NET.out_w,
            _NET.out_b,
        )
    )


def refresh(stacks: bb.BoardStacks, acc: nb.AccStacks, ply: int) -> None:
    nb.refresh_ply(
        _NET.ft_w,
        _NET.ft_b,
        _NET.psqt_w,
        stacks.mailbox,
        stacks.state,
        ply,
        acc.white_acc,
        acc.black_acc,
        acc.white_psqt,
        acc.black_psqt,
    )


def test_refresh_matches_engine_on_random_positions() -> None:
    engine = Engine(_NET)
    rng = random.Random(6)
    stacks = bb.new_stacks()
    acc = nb.new_acc_stacks()
    board = chess.Board()
    for _ in range(300):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        engine.set_position(board.fen())
        bb.set_from_board(board, stacks, 0)
        refresh(stacks, acc, 0)
        assert evaluate(stacks, acc, 0) == engine.evaluate(), board.fen()


def test_incremental_updates_match_engine_along_games() -> None:
    engine = Engine(_NET)
    rng = random.Random(7)
    moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
    for _game in range(20):
        board = chess.Board()
        engine.set_position(board.fen())
        stacks = bb.new_stacks()
        acc = nb.new_acc_stacks()
        bb.set_from_board(board, stacks, 0)
        refresh(stacks, acc, 0)
        for ply in range(80):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            count = bb.generate_moves(
                stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, ply, moves, False
            )
            ours = [int(m) for m in moves[ply, :count] if bb.move_to_uci(int(m)) == move.uci()]
            assert bb.make_move(
                stacks.pieces,
                stacks.occupied,
                stacks.mailbox,
                stacks.state,
                stacks.keys,
                ply,
                ours[0],
            )
            nb.update_ply(
                _NET.ft_w,
                _NET.ft_b,
                _NET.psqt_w,
                stacks.mailbox,
                stacks.state,
                ply,
                ours[0],
                acc.white_acc,
                acc.black_acc,
                acc.white_psqt,
                acc.black_psqt,
            )
            engine.push(move)
            board.push(move)
            assert evaluate(stacks, acc, ply + 1) == engine.evaluate(), board.fen()


def test_null_move_copy_keeps_the_evaluation() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    stacks = bb.new_stacks()
    acc = nb.new_acc_stacks()
    bb.set_from_board(board, stacks, 0)
    refresh(stacks, acc, 0)
    bb.make_null(stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, 0)
    nb.copy_ply(0, acc.white_acc, acc.black_acc, acc.white_psqt, acc.black_psqt)
    engine = Engine(_NET)
    engine.set_position(board.fen())
    engine.push_null()
    assert evaluate(stacks, acc, 1) == engine.evaluate()
