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


PERFT_POSITIONS = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [46, 2079, 89890],
    ),
]


def _stacks_for(fen: str) -> bb.BoardStacks:
    stacks = bb.new_stacks()
    bb.set_from_board(chess.Board(fen), stacks, 0)
    return stacks


def test_perft_matches_published_counts() -> None:
    moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
    for fen, counts in PERFT_POSITIONS:
        s = _stacks_for(fen)
        for depth, expected in enumerate(counts, start=1):
            got = bb.perft(s.pieces, s.occupied, s.mailbox, s.state, s.keys, 0, depth, moves)
            assert got == expected, (fen, depth, got, expected)


def _legal_uci(stacks: bb.BoardStacks, ply: int, moves: np.ndarray) -> set[str]:
    count = bb.generate_moves(
        stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, ply, moves, False
    )
    legal = set()
    for i in range(count):
        move = int(moves[ply, i])
        if bb.make_move(
            stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, ply, move
        ):
            legal.add(bb.move_to_uci(move))
    return legal


def test_legal_moves_match_python_chess_over_random_games() -> None:
    rng = random.Random(3)
    moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
    stacks = bb.new_stacks()
    positions = 0
    for _game in range(200):
        board = chess.Board()
        for _ply in range(120):
            bb.set_from_board(board, stacks, 0)
            expected = {m.uci() for m in board.legal_moves}
            assert _legal_uci(stacks, 0, moves) == expected, board.fen()
            positions += 1
            if not expected:
                break
            board.push(rng.choice(list(board.legal_moves)))
    assert positions > 5000


def test_make_move_tracks_python_chess_and_keys() -> None:
    rng = random.Random(4)
    moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
    for _game in range(50):
        board = chess.Board()
        stacks = bb.new_stacks()
        bb.set_from_board(board, stacks, 0)
        for ply in range(100):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            count = bb.generate_moves(
                stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, ply, moves, False
            )
            ours = [int(m) for m in moves[ply, :count] if bb.move_to_uci(int(m)) == move.uci()]
            assert len(ours) == 1, (board.fen(), move.uci())
            assert bb.make_move(
                stacks.pieces,
                stacks.occupied,
                stacks.mailbox,
                stacks.state,
                stacks.keys,
                ply,
                ours[0],
            )
            board.push(move)
            mirror = bb.to_board(stacks, ply + 1)
            assert mirror.board_fen() == board.board_fen(), board.fen()
            assert mirror.turn == board.turn
            assert mirror.castling_rights == board.clean_castling_rights()
            assert mirror.halfmove_clock == board.halfmove_clock
            assert stacks.keys[ply + 1] == bb.compute_key(stacks.pieces, stacks.state, ply + 1)


def test_null_move_flips_side_and_clears_en_passant() -> None:
    stacks = _stacks_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    bb.make_null(stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, 0)
    assert stacks.state[1, bb.STM] == bb.BLACK
    assert stacks.state[1, bb.EP] == -1
    assert stacks.keys[1] == bb.compute_key(stacks.pieces, stacks.state, 1)
    assert stacks.keys[1] != stacks.keys[0]
