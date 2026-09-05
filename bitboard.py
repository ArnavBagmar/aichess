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
from typing import Any, Final, cast

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
