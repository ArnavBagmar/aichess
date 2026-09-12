"""Bitboard chess board for the numba search.

The board lives in preallocated numpy stacks indexed by ply (copy-make: making a move
writes ply + 1 and unmaking is `ply -= 1`). Everything the search needs, from attack
tables to move generation, is a numba kernel over those arrays, so no Python object is
touched inside the search. python-chess is used only to parse and emit FENs at the root.

Square numbering is python-chess's: a1 = 0, h8 = 63, so tests compare directly.
Every bitboard is np.uint64. Mixing uint64 with a Python int inside a kernel promotes to
float64 silently, so integer constants below are wrapped once and reused.

Sliders use hyperbola quintessence (o ^ (o - 2r) along a masked line, and the same on
the byte-reversed board for the other direction) for files and diagonals, and a small
first-rank lookup for ranks. No magic bitboards: no opaque constants, nothing to explain.
"""

from collections.abc import Callable
from typing import Any, Final, NamedTuple, cast

import chess
import numpy as np
import numpy.typing as npt
from numba import njit

MAX_PLY: Final = 256
MAX_MOVES: Final = 256

WHITE: Final = 0
BLACK: Final = 1

# Piece types; piece codes are type + 6 * colour, and 12 marks an empty square.
PAWN: Final = 0
KNIGHT: Final = 1
BISHOP: Final = 2
ROOK: Final = 3
QUEEN: Final = 4
KING: Final = 5
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
EMPTY: Final = 12

# state[ply, :] layout.
STM: Final = 0
CASTLING: Final = 1
EP: Final = 2
HALFMOVE: Final = 3
WKING: Final = 4
BKING: Final = 5
STATE_SIZE: Final = 6

CASTLE_WK: Final = 1
CASTLE_WQ: Final = 2
CASTLE_BK: Final = 4
CASTLE_BQ: Final = 8

FLAG_CAPTURE: Final = 1
FLAG_EP: Final = 2
FLAG_CASTLE: Final = 4
FLAG_DOUBLE: Final = 8

ZERO: Final = np.uint64(0)
ONE: Final = np.uint64(1)
RANK_1: Final = np.uint64(0x00000000000000FF)
RANK_3: Final = np.uint64(0x0000000000FF0000)
RANK_6: Final = np.uint64(0x0000FF0000000000)
RANK_8: Final = np.uint64(0xFF00000000000000)


def _jit[F: Callable[..., Any]](function: F) -> F:
    """Typed facade over numba.njit so call sites keep their signatures for mypy."""
    return cast("F", njit(cache=False)(function))


# --- tables, built in plain Python at import ---------------------------------------------


def _leaper(deltas: list[tuple[int, int]]) -> npt.NDArray[np.uint64]:
    table = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        mask = 0
        for dr, df in deltas:
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                mask |= 1 << (r * 8 + f)
        table[sq] = mask
    return table


KNIGHT_ATTACKS: Final = _leaper(
    [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
)
KING_ATTACKS: Final = _leaper(
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
)
PAWN_ATTACKS: Final = np.stack([_leaper([(1, -1), (1, 1)]), _leaper([(-1, -1), (-1, 1)])])


def _line_masks() -> tuple[npt.NDArray[np.uint64], ...]:
    """File, diagonal and anti-diagonal through each square, the square itself excluded."""
    file_mask = np.zeros(64, dtype=np.uint64)
    diag_mask = np.zeros(64, dtype=np.uint64)
    anti_mask = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        for other in range(64):
            if other == sq:
                continue
            r, f = divmod(other, 8)
            if f == file:
                file_mask[sq] |= np.uint64(1 << other)
            if r - f == rank - file:
                diag_mask[sq] |= np.uint64(1 << other)
            if r + f == rank + file:
                anti_mask[sq] |= np.uint64(1 << other)
    return file_mask, diag_mask, anti_mask


FILE_MASK, DIAG_MASK, ANTI_MASK = _line_masks()


def _rank_attacks() -> npt.NDArray[np.uint64]:
    """RANK_ATTACKS[sq, inner]: attacks along the rank for its six inner occupancy bits."""
    table = np.zeros((64, 64), dtype=np.uint64)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        for inner in range(64):
            occ = inner << 1  # files b..g; the edge files never block anything beyond them
            attacks = 0
            for step in (1, -1):
                f = file + step
                while 0 <= f < 8:
                    attacks |= 1 << f
                    if occ & (1 << f):
                        break
                    f += step
            table[sq, inner] = attacks << (rank * 8)
    return table


RANK_ATTACKS: Final = _rank_attacks()

# Castling rights that survive a piece leaving or arriving on each square.
CASTLE_MASK: Final = np.full(64, 15, dtype=np.int32)
CASTLE_MASK[chess.E1] = 15 & ~(CASTLE_WK | CASTLE_WQ)
CASTLE_MASK[chess.H1] = 15 & ~CASTLE_WK
CASTLE_MASK[chess.A1] = 15 & ~CASTLE_WQ
CASTLE_MASK[chess.E8] = 15 & ~(CASTLE_BK | CASTLE_BQ)
CASTLE_MASK[chess.H8] = 15 & ~CASTLE_BK
CASTLE_MASK[chess.A8] = 15 & ~CASTLE_BQ

_rng = np.random.default_rng(20260905)
_U64_MAX = np.iinfo(np.uint64).max
ZOBRIST_PIECE: Final = _rng.integers(0, _U64_MAX, size=(12, 64), dtype=np.uint64, endpoint=True)
ZOBRIST_CASTLE: Final = _rng.integers(0, _U64_MAX, size=16, dtype=np.uint64, endpoint=True)
ZOBRIST_EP: Final = _rng.integers(0, _U64_MAX, size=8, dtype=np.uint64, endpoint=True)
ZOBRIST_SIDE: Final = np.uint64(_rng.integers(0, _U64_MAX, dtype=np.uint64, endpoint=True))


# --- bit helpers -------------------------------------------------------------------------


@_jit
def bit(sq: int) -> np.uint64:
    return ONE << np.uint64(sq)


@_jit
def popcount(x: np.uint64) -> int:
    x = x - ((x >> ONE) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@_jit
def lsb(x: np.uint64) -> int:
    """Index of the lowest set bit; x must be non-zero."""
    return popcount((x & (~x + ONE)) - ONE)


@_jit
def byteswap(x: np.uint64) -> np.uint64:
    x = (x >> np.uint64(32)) | (x << np.uint64(32))
    x = ((x & np.uint64(0xFFFF0000FFFF0000)) >> np.uint64(16)) | (
        (x & np.uint64(0x0000FFFF0000FFFF)) << np.uint64(16)
    )
    x = ((x & np.uint64(0xFF00FF00FF00FF00)) >> np.uint64(8)) | (
        (x & np.uint64(0x00FF00FF00FF00FF)) << np.uint64(8)
    )
    return x


@_jit
def _line_attacks(occ: np.uint64, sq: int, mask: np.uint64) -> np.uint64:
    """Hyperbola quintessence along one line (file, diagonal or anti-diagonal)."""
    o = occ & mask
    r = bit(sq)
    forward = o - (r << ONE)
    backward = byteswap(byteswap(o) - (byteswap(r) << ONE))
    return (forward ^ backward) & mask


@_jit
def bishop_attacks(sq: int, occ: np.uint64) -> np.uint64:
    return _line_attacks(occ, sq, DIAG_MASK[sq]) | _line_attacks(occ, sq, ANTI_MASK[sq])


@_jit
def rook_attacks(sq: int, occ: np.uint64) -> np.uint64:
    rank_shift = np.uint64((sq >> 3) * 8 + 1)
    inner = int((occ >> rank_shift) & np.uint64(63))
    return np.uint64(_line_attacks(occ, sq, FILE_MASK[sq]) | RANK_ATTACKS[sq, inner])


@_jit
def queen_attacks(sq: int, occ: np.uint64) -> np.uint64:
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


@_jit
def is_attacked(pieces: npt.NDArray[np.uint64], sq: int, by: int, occ: np.uint64) -> bool:
    """Whether colour `by` attacks `sq` given occupancy `occ`; `pieces` is one ply's row."""
    base = by * 6
    if PAWN_ATTACKS[by ^ 1, sq] & pieces[base + PAWN] != ZERO:
        return True
    if KNIGHT_ATTACKS[sq] & pieces[base + KNIGHT] != ZERO:
        return True
    if KING_ATTACKS[sq] & pieces[base + KING] != ZERO:
        return True
    diagonal = pieces[base + BISHOP] | pieces[base + QUEEN]
    if bishop_attacks(sq, occ) & diagonal != ZERO:
        return True
    straight = pieces[base + ROOK] | pieces[base + QUEEN]
    return bool(rook_attacks(sq, occ) & straight != ZERO)


@_jit
def attackers_to(pieces: npt.NDArray[np.uint64], sq: int, occ: np.uint64) -> np.uint64:
    """Every piece of either colour attacking `sq` under occupancy `occ`."""
    result = PAWN_ATTACKS[BLACK, sq] & pieces[WP]
    result |= PAWN_ATTACKS[WHITE, sq] & pieces[BP]
    result |= KNIGHT_ATTACKS[sq] & (pieces[WN] | pieces[BN])
    result |= KING_ATTACKS[sq] & (pieces[WK] | pieces[BK])
    result |= bishop_attacks(sq, occ) & (pieces[WB] | pieces[BB] | pieces[WQ] | pieces[BQ])
    result |= rook_attacks(sq, occ) & (pieces[WR] | pieces[BR] | pieces[WQ] | pieces[BQ])
    return np.uint64(result & occ)


# --- moves -------------------------------------------------------------------------------


def encode_move(frm: int, to: int, promotion: int = 0, flags: int = 0) -> int:
    """from | to << 6 | promotion << 12 | flags << 15; promotion is a piece type or 0."""
    return frm | (to << 6) | (promotion << 12) | (flags << 15)


def move_from(move: int) -> int:
    return move & 63


def move_to(move: int) -> int:
    return (move >> 6) & 63


def move_promotion(move: int) -> int:
    return (move >> 12) & 7


def move_flags(move: int) -> int:
    return move >> 15


def move_to_chess(move: int) -> chess.Move:
    promotion = move_promotion(move)
    return chess.Move(move_from(move), move_to(move), promotion + 1 if promotion else None)


def move_to_uci(move: int) -> str:
    return move_to_chess(move).uci()


def piece_code(piece: chess.Piece) -> int:
    return piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)


# --- board stacks and FEN -----------------------------------------------------------------


class BoardStacks(NamedTuple):
    """One board per ply. Making a move writes ply + 1; unmaking is `ply -= 1`."""

    pieces: npt.NDArray[np.uint64]  # [MAX_PLY, 12] one bitboard per piece code
    occupied: npt.NDArray[np.uint64]  # [MAX_PLY, 3] white, black, both
    mailbox: npt.NDArray[np.int8]  # [MAX_PLY, 64] piece code per square, EMPTY if none
    state: npt.NDArray[np.int32]  # [MAX_PLY, STATE_SIZE]
    keys: npt.NDArray[np.uint64]  # [MAX_PLY] zobrist key


def new_stacks() -> BoardStacks:
    return BoardStacks(
        np.zeros((MAX_PLY, 12), dtype=np.uint64),
        np.zeros((MAX_PLY, 3), dtype=np.uint64),
        np.full((MAX_PLY, 64), EMPTY, dtype=np.int8),
        np.zeros((MAX_PLY, STATE_SIZE), dtype=np.int32),
        np.zeros(MAX_PLY, dtype=np.uint64),
    )


def set_from_board(board: chess.Board, stacks: BoardStacks, ply: int) -> None:
    """Fill ply `ply` of the stacks from a python-chess board."""
    pieces, occupied, mailbox, state, keys = stacks
    pieces[ply, :] = 0
    mailbox[ply, :] = EMPTY
    for sq, piece in board.piece_map().items():
        code = piece_code(piece)
        pieces[ply, code] |= np.uint64(1 << sq)
        mailbox[ply, sq] = code
    occupied[ply, WHITE] = np.uint64(board.occupied_co[chess.WHITE])
    occupied[ply, BLACK] = np.uint64(board.occupied_co[chess.BLACK])
    occupied[ply, 2] = np.uint64(board.occupied)
    rights = board.clean_castling_rights()
    castling = 0
    if rights & chess.BB_H1:
        castling |= CASTLE_WK
    if rights & chess.BB_A1:
        castling |= CASTLE_WQ
    if rights & chess.BB_H8:
        castling |= CASTLE_BK
    if rights & chess.BB_A8:
        castling |= CASTLE_BQ
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("position has no king for one side")
    state[ply, STM] = WHITE if board.turn == chess.WHITE else BLACK
    state[ply, CASTLING] = castling
    state[ply, EP] = board.ep_square if board.ep_square is not None else -1
    state[ply, HALFMOVE] = board.halfmove_clock
    state[ply, WKING] = white_king
    state[ply, BKING] = black_king
    keys[ply] = compute_key(pieces, state, ply)


def to_board(stacks: BoardStacks, ply: int) -> chess.Board:
    """A python-chess board for ply `ply`, for tests and debugging."""
    board = chess.Board(None)
    for sq in range(64):
        code = int(stacks.mailbox[ply, sq])
        if code != EMPTY:
            colour = chess.WHITE if code < 6 else chess.BLACK
            board.set_piece_at(sq, chess.Piece(code % 6 + 1, colour))
    board.turn = chess.WHITE if stacks.state[ply, STM] == WHITE else chess.BLACK
    castling = int(stacks.state[ply, CASTLING])
    rights = 0
    if castling & CASTLE_WK:
        rights |= chess.BB_H1
    if castling & CASTLE_WQ:
        rights |= chess.BB_A1
    if castling & CASTLE_BK:
        rights |= chess.BB_H8
    if castling & CASTLE_BQ:
        rights |= chess.BB_A8
    board.castling_rights = rights
    ep = int(stacks.state[ply, EP])
    board.ep_square = ep if ep >= 0 else None
    board.halfmove_clock = int(stacks.state[ply, HALFMOVE])
    return board


@_jit
def compute_key(
    pieces: npt.NDArray[np.uint64], state: npt.NDArray[np.int32], ply: int
) -> np.uint64:
    key = ZERO
    for code in range(12):
        bbits = pieces[ply, code]
        while bbits != ZERO:
            sq = lsb(bbits)
            bbits &= bbits - ONE
            key ^= ZOBRIST_PIECE[code, sq]
    key ^= ZOBRIST_CASTLE[state[ply, CASTLING]]
    if state[ply, EP] >= 0:
        key ^= ZOBRIST_EP[state[ply, EP] & 7]
    if state[ply, STM] == BLACK:
        key ^= ZOBRIST_SIDE
    return np.uint64(key)


# --- move generation ----------------------------------------------------------------------


@_jit
def _add_pawn_moves(
    moves: npt.NDArray[np.int32],
    ply: int,
    n: int,
    frm: int,
    to: int,
    flags: int,
    promote: bool,
    queen_only: bool,
) -> int:
    if promote:
        moves[ply, n] = frm | (to << 6) | (QUEEN << 12) | (flags << 15)
        n += 1
        if not queen_only:
            for piece in (KNIGHT, BISHOP, ROOK):
                moves[ply, n] = frm | (to << 6) | (piece << 12) | (flags << 15)
                n += 1
    else:
        moves[ply, n] = frm | (to << 6) | (flags << 15)
        n += 1
    return n


@_jit
def generate_moves(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    ply: int,
    moves: npt.NDArray[np.int32],
    tactical_only: bool,
) -> int:
    """Pseudo-legal moves for ply `ply` into moves[ply]; returns the count.

    Legality (own king left in check) is decided by make_move. With tactical_only, only
    captures and queen promotions are generated, which is what quiescence searches.
    Castling checks the usual rule here: never out of, through or into check.
    """
    n = 0
    stm = state[ply, STM]
    own = occupied[ply, stm]
    enemy = occupied[ply, stm ^ 1]
    occ = occupied[ply, 2]
    empty = ~occ
    base = stm * 6
    ep = state[ply, EP]
    ep_bit = bit(ep) if ep >= 0 else ZERO

    pawns = pieces[ply, base + PAWN]
    if stm == WHITE:
        last_rank = RANK_8
        single = (pawns << np.uint64(8)) & empty
        double = ((single & RANK_3) << np.uint64(8)) & empty
        back = -8
    else:
        last_rank = RANK_1
        single = (pawns >> np.uint64(8)) & empty
        double = ((single & RANK_6) >> np.uint64(8)) & empty
        back = 8

    targets = single & last_rank if tactical_only else single
    while targets != ZERO:
        to = lsb(targets)
        targets &= targets - ONE
        promote = (bit(to) & last_rank) != ZERO
        n = _add_pawn_moves(moves, ply, n, to + back, to, 0, promote, tactical_only)
    if not tactical_only:
        while double != ZERO:
            to = lsb(double)
            double &= double - ONE
            moves[ply, n] = (to + 2 * back) | (to << 6) | (FLAG_DOUBLE << 15)
            n += 1
    remaining = pawns
    while remaining != ZERO:
        frm = lsb(remaining)
        remaining &= remaining - ONE
        attacks = PAWN_ATTACKS[stm, frm]
        captures = attacks & enemy
        while captures != ZERO:
            to = lsb(captures)
            captures &= captures - ONE
            promote = (bit(to) & last_rank) != ZERO
            n = _add_pawn_moves(moves, ply, n, frm, to, FLAG_CAPTURE, promote, tactical_only)
        if attacks & ep_bit != ZERO:
            moves[ply, n] = frm | (ep << 6) | ((FLAG_CAPTURE | FLAG_EP) << 15)
            n += 1

    for piece in range(KNIGHT, KING + 1):
        bbits = pieces[ply, base + piece]
        while bbits != ZERO:
            frm = lsb(bbits)
            bbits &= bbits - ONE
            if piece == KNIGHT:
                attacks = KNIGHT_ATTACKS[frm]
            elif piece == BISHOP:
                attacks = bishop_attacks(frm, occ)
            elif piece == ROOK:
                attacks = rook_attacks(frm, occ)
            elif piece == QUEEN:
                attacks = queen_attacks(frm, occ)
            else:
                attacks = KING_ATTACKS[frm]
            attacks &= ~own
            if tactical_only:
                attacks &= enemy
            while attacks != ZERO:
                to = lsb(attacks)
                attacks &= attacks - ONE
                flags = FLAG_CAPTURE if (bit(to) & enemy) != ZERO else 0
                moves[ply, n] = frm | (to << 6) | (flags << 15)
                n += 1

    if not tactical_only:
        rights = state[ply, CASTLING]
        if stm == WHITE:
            king_sq, kingside, queenside, rooks = 4, CASTLE_WK, CASTLE_WQ, pieces[ply, WR]
        else:
            king_sq, kingside, queenside, rooks = 60, CASTLE_BK, CASTLE_BQ, pieces[ply, BR]
        if rights & (kingside | queenside) and not is_attacked(pieces[ply], king_sq, stm ^ 1, occ):
            if (
                rights & kingside
                and rooks & bit(king_sq + 3) != ZERO
                and occ & (bit(king_sq + 1) | bit(king_sq + 2)) == ZERO
                and not is_attacked(pieces[ply], king_sq + 1, stm ^ 1, occ)
                and not is_attacked(pieces[ply], king_sq + 2, stm ^ 1, occ)
            ):
                moves[ply, n] = king_sq | ((king_sq + 2) << 6) | (FLAG_CASTLE << 15)
                n += 1
            if (
                rights & queenside
                and rooks & bit(king_sq - 4) != ZERO
                and occ & (bit(king_sq - 1) | bit(king_sq - 2) | bit(king_sq - 3)) == ZERO
                and not is_attacked(pieces[ply], king_sq - 1, stm ^ 1, occ)
                and not is_attacked(pieces[ply], king_sq - 2, stm ^ 1, occ)
            ):
                moves[ply, n] = king_sq | ((king_sq - 2) << 6) | (FLAG_CASTLE << 15)
                n += 1
    return n


# --- make ---------------------------------------------------------------------------------


@_jit
def _copy_ply(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    ply: int,
) -> None:
    nxt = ply + 1
    for i in range(12):
        pieces[nxt, i] = pieces[ply, i]
    for i in range(3):
        occupied[nxt, i] = occupied[ply, i]
    for i in range(64):
        mailbox[nxt, i] = mailbox[ply, i]
    for i in range(STATE_SIZE):
        state[nxt, i] = state[ply, i]


@_jit
def _remove_piece(
    pieces: npt.NDArray[np.uint64], mailbox: npt.NDArray[np.int8], ply: int, sq: int, code: int
) -> np.uint64:
    pieces[ply, code] &= ~bit(sq)
    mailbox[ply, sq] = EMPTY
    return np.uint64(ZOBRIST_PIECE[code, sq])


@_jit
def _put_piece(
    pieces: npt.NDArray[np.uint64], mailbox: npt.NDArray[np.int8], ply: int, sq: int, code: int
) -> np.uint64:
    pieces[ply, code] |= bit(sq)
    mailbox[ply, sq] = code
    return np.uint64(ZOBRIST_PIECE[code, sq])


@_jit
def make_move(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    keys: npt.NDArray[np.uint64],
    ply: int,
    move: int,
) -> bool:
    """Write ply + 1 as the position after `move`.

    Returns False if the move leaves the mover's king in check; ply + 1 is then garbage
    and must not be entered.
    """
    _copy_ply(pieces, occupied, mailbox, state, ply)
    nxt = ply + 1
    stm = state[ply, STM]
    frm = move & 63
    to = (move >> 6) & 63
    promotion = (move >> 12) & 7
    flags = move >> 15
    mover = mailbox[ply, frm]
    key = keys[ply]

    key ^= _remove_piece(pieces, mailbox, nxt, frm, mover)
    if flags & FLAG_EP:
        captured_sq = to - 8 if stm == WHITE else to + 8
        key ^= _remove_piece(pieces, mailbox, nxt, captured_sq, mailbox[ply, captured_sq])
    elif flags & FLAG_CAPTURE:
        key ^= _remove_piece(pieces, mailbox, nxt, to, mailbox[ply, to])
    landed = mover if promotion == 0 else promotion + stm * 6
    key ^= _put_piece(pieces, mailbox, nxt, to, landed)
    if flags & FLAG_CASTLE:
        rook = ROOK + stm * 6
        if to > frm:
            key ^= _remove_piece(pieces, mailbox, nxt, frm + 3, rook)
            key ^= _put_piece(pieces, mailbox, nxt, frm + 1, rook)
        else:
            key ^= _remove_piece(pieces, mailbox, nxt, frm - 4, rook)
            key ^= _put_piece(pieces, mailbox, nxt, frm - 1, rook)

    old_rights = state[ply, CASTLING]
    new_rights = old_rights & CASTLE_MASK[frm] & CASTLE_MASK[to]
    if new_rights != old_rights:
        key ^= ZOBRIST_CASTLE[old_rights] ^ ZOBRIST_CASTLE[new_rights]
    state[nxt, CASTLING] = new_rights

    if state[ply, EP] >= 0:
        key ^= ZOBRIST_EP[state[ply, EP] & 7]
    if flags & FLAG_DOUBLE:
        ep_sq = (frm + to) >> 1
        state[nxt, EP] = ep_sq
        key ^= ZOBRIST_EP[ep_sq & 7]
    else:
        state[nxt, EP] = -1

    if mover % 6 == PAWN or flags & FLAG_CAPTURE:
        state[nxt, HALFMOVE] = 0
    else:
        state[nxt, HALFMOVE] = state[ply, HALFMOVE] + 1
    if mover % 6 == KING:
        state[nxt, WKING + stm] = to
    state[nxt, STM] = stm ^ 1
    key ^= ZOBRIST_SIDE
    keys[nxt] = key

    white = ZERO
    for code in range(6):
        white |= pieces[nxt, code]
    black = ZERO
    for code in range(6, 12):
        black |= pieces[nxt, code]
    occupied[nxt, WHITE] = white
    occupied[nxt, BLACK] = black
    occupied[nxt, 2] = white | black
    return not is_attacked(pieces[nxt], state[nxt, WKING + stm], stm ^ 1, white | black)


@_jit
def copy_position(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    keys: npt.NDArray[np.uint64],
    ply: int,
) -> None:
    """Duplicate the position at `ply` into ply + 1, side to move and all.

    The search uses this to run a sub-search on the same position from a scratch
    ply, since every per-ply table of the ply itself is in use by the move loop.
    """
    _copy_ply(pieces, occupied, mailbox, state, ply)
    keys[ply + 1] = keys[ply]


@_jit
def make_null(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    keys: npt.NDArray[np.uint64],
    ply: int,
) -> None:
    """Pass the turn into ply + 1."""
    _copy_ply(pieces, occupied, mailbox, state, ply)
    nxt = ply + 1
    key = keys[ply]
    if state[ply, EP] >= 0:
        key ^= ZOBRIST_EP[state[ply, EP] & 7]
    state[nxt, EP] = -1
    state[nxt, HALFMOVE] = state[ply, HALFMOVE] + 1
    state[nxt, STM] = state[ply, STM] ^ 1
    keys[nxt] = key ^ ZOBRIST_SIDE


@_jit
def perft(
    pieces: npt.NDArray[np.uint64],
    occupied: npt.NDArray[np.uint64],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    keys: npt.NDArray[np.uint64],
    ply: int,
    depth: int,
    moves: npt.NDArray[np.int32],
) -> int:
    """Leaf count at `depth`, the standard move generator test."""
    if depth == 0:
        return 1
    count = generate_moves(pieces, occupied, mailbox, state, ply, moves, False)
    total = 0
    for i in range(count):
        if make_move(pieces, occupied, mailbox, state, keys, ply, moves[ply, i]):
            total += perft(pieces, occupied, mailbox, state, keys, ply + 1, depth - 1, moves)
    return total
