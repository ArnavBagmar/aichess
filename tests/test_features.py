"""Unit tests for HalfKAv2_hm feature extraction.

The hand-computed indices follow nnue-pytorch's halfka_v2_hm.py exactly:
index = orient(sq) + 64 * piece_plane + 768 * KING_BUCKETS[orient(king_sq)].
"""

import random

import chess
import pytest

from nnue_arch import MAX_ACTIVE_FEATURES, NUM_FEATURES
from nnue_features import active_features, feature_index, move_deltas


def test_white_pawn_from_white_king_on_e1() -> None:
    # King e1 (file >= 4, no mirror) -> bucket KING_BUCKETS[4] = 31. White pawn e2,
    # own pawn plane 0: index = 12 + 64*0 + 768*31.
    piece = chess.Piece(chess.PAWN, chess.WHITE)
    assert feature_index(True, chess.E1, chess.E2, piece) == 12 + 768 * 31


def test_white_pawn_from_black_king_on_e8() -> None:
    # Black pov: e8 flips to 4 -> bucket 31. White pawn e2 flips to 52, enemy plane 1.
    piece = chess.Piece(chess.PAWN, chess.WHITE)
    assert feature_index(False, chess.E8, chess.E2, piece) == 52 + 64 + 768 * 31


def test_mirrored_king_on_a1() -> None:
    # King a1 mirrors to h1 (7) -> bucket 28. Knight b1 mirrors to g1 (6), own plane 2.
    piece = chess.Piece(chess.KNIGHT, chess.WHITE)
    assert feature_index(True, chess.A1, chess.B1, piece) == 6 + 64 * 2 + 768 * 28


def test_startpos_perspectives_agree() -> None:
    # The start position is symmetric, so both perspectives see identical feature sets.
    board = chess.Board()
    assert sorted(active_features(board, True)) == sorted(active_features(board, False))


def test_feature_count_and_range() -> None:
    board = chess.Board()
    for pov in (True, False):
        features = active_features(board, pov)
        assert len(features) == 32
        assert len(set(features)) == 32
        assert all(0 <= f < NUM_FEATURES for f in features)
        assert len(features) <= MAX_ACTIVE_FEATURES


def test_own_king_move_returns_none() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    king_move = chess.Move.from_uci("e1d1")
    assert move_deltas(board, king_move, True) is None
    assert move_deltas(board, king_move, False) is not None


@pytest.mark.parametrize(
    "fen,uci",
    [
        # Quiet move, capture, promotion, capture-promotion, en passant, castling.
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
        ("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7g8q"),
        ("4kn2/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7f8n"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8c8"),
        ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", "e8g8"),
    ],
)
def test_deltas_match_recomputation(fen: str, uci: str) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    for pov in (True, False):
        deltas = move_deltas(board, move, pov)
        before = sorted(active_features(board, pov))
        board.push(move)
        after = sorted(active_features(board, pov))
        board.pop()
        if deltas is None:
            # Own king moved; the only correct handling is a rebuild, nothing to check.
            mover = board.piece_at(move.from_square)
            assert mover is not None and mover.piece_type == chess.KING
            continue
        added, removed = deltas
        updated = sorted(set(before) - set(removed) | set(added))
        assert updated == after, f"{fen} {uci} pov={pov}"


def test_deltas_match_recomputation_random_games() -> None:
    rng = random.Random(7)
    for _ in range(25):
        board = chess.Board()
        for _ply in range(120):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            for pov in (True, False):
                deltas = move_deltas(board, move, pov)
                if deltas is None:
                    continue
                added, removed = deltas
                before = sorted(active_features(board, pov))
                board.push(move)
                after = sorted(active_features(board, pov))
                board.pop()
                updated = sorted(set(before) - set(removed) | set(added))
                assert updated == after, f"{board.fen()} {move.uci()} pov={pov}"
            board.push(move)
