"""The bitboard module against python-chess, which is the oracle for every rule."""

import random

import chess
import numpy as np

import bitboard as bb


def test_bit_helpers() -> None:
    assert bb.popcount(np.uint64(0)) == 0
    assert bb.popcount(np.uint64(0xFFFFFFFFFFFFFFFF)) == 64
    assert bb.lsb(np.uint64(1) << np.uint64(63)) == 63
    assert bb.lsb(np.uint64(0b1011000)) == 3
    assert bb.bit(10) == np.uint64(1 << 10)


def test_leaper_tables_match_python_chess() -> None:
    for sq in range(64):
        assert int(bb.KNIGHT_ATTACKS[sq]) == chess.BB_KNIGHT_ATTACKS[sq]
        assert int(bb.KING_ATTACKS[sq]) == chess.BB_KING_ATTACKS[sq]
        assert int(bb.PAWN_ATTACKS[bb.WHITE, sq]) == chess.BB_PAWN_ATTACKS[chess.WHITE][sq]
        assert int(bb.PAWN_ATTACKS[bb.BLACK, sq]) == chess.BB_PAWN_ATTACKS[chess.BLACK][sq]


def test_slider_attacks_match_python_chess_on_random_occupancy() -> None:
    rng = random.Random(1)
    for _ in range(2000):
        occ = rng.getrandbits(64)
        sq = rng.randrange(64)
        u_occ = np.uint64(occ)
        diag = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
        line = (
            chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
            | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
        )
        assert int(bb.bishop_attacks(sq, u_occ)) == diag, (sq, hex(occ))
        assert int(bb.rook_attacks(sq, u_occ)) == line, (sq, hex(occ))
        assert int(bb.queen_attacks(sq, u_occ)) == diag | line


def test_is_attacked_matches_python_chess() -> None:
    rng = random.Random(2)
    board = chess.Board()
    for _ in range(300):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        pieces = np.zeros((1, 12), dtype=np.uint64)
        for sq, piece in board.piece_map().items():
            pieces[0, bb.piece_code(piece)] |= np.uint64(1 << sq)
        occ = np.uint64(board.occupied)
        for sq in range(64):
            for color in (chess.WHITE, chess.BLACK):
                expected = board.is_attacked_by(color, sq)
                got = bb.is_attacked(pieces[0], sq, 0 if color else 1, occ)
                assert got == expected, (board.fen(), sq, color)
            expected_mask = board.attackers_mask(chess.WHITE, sq) | board.attackers_mask(
                chess.BLACK, sq
            )
            assert int(bb.attackers_to(pieces[0], sq, occ)) == expected_mask


def test_move_encoding_round_trips() -> None:
    move = bb.encode_move(12, 28, 0, bb.FLAG_DOUBLE)
    assert bb.move_from(move) == 12
    assert bb.move_to(move) == 28
    assert bb.move_promotion(move) == 0
    assert bb.move_flags(move) == bb.FLAG_DOUBLE
    assert bb.move_to_uci(move) == "e2e4"
    promo = bb.encode_move(52, 60, bb.QUEEN, 0)
    assert bb.move_to_uci(promo) == "e7e8q"
