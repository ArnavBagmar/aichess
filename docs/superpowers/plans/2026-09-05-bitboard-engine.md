# Bitboard Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace python-chess inside the search with numba bitboard kernels (move generation, copy-make, NNUE update, alpha-beta) so the same search runs about ten times faster.

**Architecture:** `bitboard.py` holds the board arrays, attack tables, move generation and make-move; `nnue_bitboard.py` computes HalfKAv2_hm features and accumulators on those arrays; `search_kernel.py` is one recursive `@njit` alpha-beta (quiescence is its `depth <= 0` branch) over preallocated numpy stacks; `search.py` becomes a thin Python `Searcher` that parses the FEN with python-chess, runs iterative deepening and aspiration, and validates the returned move. python-chess remains the oracle in tests (perft, differential move generation, evaluation parity with `nnue_engine.Engine`).

**Tech Stack:** Python 3.12, numpy 2.5, numba 0.67, python-chess 1.11 (root and tests only). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-bitboard-engine-design.md`

## Global Constraints

- `uv run ruff check .` and `uv run mypy` (strict) pass before every commit. Line length 100.
- Only `chess`, `numpy`, `numba` are imported by shipped modules. All kernels `njit(cache=False)`.
- Do not edit `harness/`. Do not edit `nnue_arch.py`, `nnue_net.py`, `nnue_features.py`. `nnue_engine.py` stays until Task 7 as the parity oracle.
- Square numbering is python-chess's: a1 = 0, b1 = 1, h8 = 63. Piece codes: white P N B R Q K = 0..5, black = 6..11, empty = 12.
- Scores are integers in 1/32 cp, positive for the side to move. `MATE = 1_000_000`. Every search constant is copied from `search.py` at 74a7d63 unchanged.
- Every bitboard value is `np.uint64`; never mix with Python ints inside a kernel without `np.uint64(...)`. Mixing uint64 and int64 in numba silently promotes to float64.
- Compile time of `import agent` is measured from Task 4 on and must stay under 30 s on this machine.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013Sah6pF8HY5E5xtY5Vd8ce
  ```
- The GPU is training net2 in the background; never launch more than one bench at a time and keep `--workers 3`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `bitboard.py` (create) | constants, attack tables, bit helpers, zobrist tables, FEN in/out, `generate_moves`, `make_move`, `make_null`, `is_attacked`, `attackers_to`, `perft` |
| `nnue_bitboard.py` (create) | `feature_index`, `refresh_ply`, `update_ply`, `copy_ply`, `evaluate` (with `_forward` moved from `nnue_engine.py`) |
| `search_kernel.py` (create) | constants, `see`, TT pack/unpack, `search` kernel, `now` clock |
| `search.py` (rewrite) | `Searcher`, `budget_ms`, aspiration helpers, move conversion |
| `tests/test_bitboard.py` (create) | attacks, perft, differential movegen, keys |
| `tests/test_nnue_bitboard.py` (create) | evaluation parity with `nnue_engine.Engine` |
| `tests/test_search.py` (rewrite) | behavioural search tests on the new `Searcher` |
| `tests/test_import_time.py` (create) | fresh-interpreter import budget |
| `tools/nps.py` (modify) | speed probe on the new `Searcher` |

---

### Task 1: Board arrays, attack tables and bit helpers

**Files:**
- Create: `bitboard.py`
- Test: `tests/test_bitboard.py`

**Interfaces:**
- Produces: constants `MAX_PLY`, `MAX_MOVES`, `WHITE`, `BLACK`, piece codes `WP..BK`, `EMPTY`, piece types `PAWN..KING`, state indices `STM, CASTLING, EP, HALFMOVE, WKING, BKING`, castling bits `CASTLE_WK..CASTLE_BQ`, move flags `FLAG_CAPTURE, FLAG_EP, FLAG_CASTLE, FLAG_DOUBLE`; kernels `bit(sq) -> uint64`, `popcount(bb) -> int`, `lsb(bb) -> int`, `bishop_attacks(sq, occ)`, `rook_attacks(sq, occ)`, `queen_attacks(sq, occ)`, `is_attacked(pieces_row, sq, by, occ) -> bool`, `attackers_to(pieces_row, sq, occ) -> uint64`; Python `encode_move`, `move_from`, `move_to`, `move_promotion`, `move_flags`, `move_to_uci`, `move_to_chess`.

- [ ] **Step 1: Write the failing tests** (`tests/test_bitboard.py`)

```python
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
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_bitboard.py -q` fails with `ModuleNotFoundError: No module named 'bitboard'`.

- [ ] **Step 3: Write `bitboard.py` part 1**

```python
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
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
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
    return _line_attacks(occ, sq, FILE_MASK[sq]) | RANK_ATTACKS[sq, inner]


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
    return rook_attacks(sq, occ) & straight != ZERO


@_jit
def attackers_to(pieces: npt.NDArray[np.uint64], sq: int, occ: np.uint64) -> np.uint64:
    """Every piece of either colour attacking `sq` under occupancy `occ`."""
    result = PAWN_ATTACKS[BLACK, sq] & pieces[WP]
    result |= PAWN_ATTACKS[WHITE, sq] & pieces[BP]
    result |= KNIGHT_ATTACKS[sq] & (pieces[WN] | pieces[BN])
    result |= KING_ATTACKS[sq] & (pieces[WK] | pieces[BK])
    result |= bishop_attacks(sq, occ) & (pieces[WB] | pieces[BB] | pieces[WQ] | pieces[BQ])
    result |= rook_attacks(sq, occ) & (pieces[WR] | pieces[BR] | pieces[WQ] | pieces[BQ])
    return result & occ


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
```

- [ ] **Step 4: Run**: `uv run pytest tests/test_bitboard.py -q`, expect 5 passed. If `rook_attacks` disagrees, the rank lookup index must be the six inner bits `(occ >> (rank*8 + 1)) & 63`.

- [ ] **Step 5: Lint, types, commit** (add `"bitboard.py"` to mypy `files` in `pyproject.toml`):

```bash
uv run ruff check . && uv run mypy
git add bitboard.py tests/test_bitboard.py pyproject.toml
git commit -m "feat(bitboard): attack tables, bit helpers and move encoding"
```

---

### Task 2: FEN, zobrist keys, move generation and copy-make

**Files:**
- Modify: `bitboard.py` (append)
- Test: `tests/test_bitboard.py` (append)

**Interfaces:**
- Produces: `BoardStacks` NamedTuple (`pieces uint64[MAX_PLY,12]`, `occupied uint64[MAX_PLY,3]`, `mailbox int8[MAX_PLY,64]`, `state int32[MAX_PLY,6]`, `keys uint64[MAX_PLY]`), `new_stacks()`, `set_from_board(board, stacks, ply)`, `to_board(stacks, ply) -> chess.Board`; kernels `compute_key(pieces, state, ply) -> uint64`, `generate_moves(pieces, occupied, mailbox, state, ply, moves, tactical_only) -> int`, `make_move(pieces, occupied, mailbox, state, keys, ply, move) -> bool` (False and ply+1 dirty when illegal), `make_null(pieces, occupied, mailbox, state, keys, ply)`, `perft(pieces, occupied, mailbox, state, keys, ply, depth, moves) -> int`.

- [ ] **Step 1: Append the failing tests**

```python
PERFT_POSITIONS = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890]),
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
        ply = 0
        for _ in range(100):
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
                stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, ply,
                ours[0],
            )
            board.push(move)
            ply += 1
            mirror = bb.to_board(stacks, ply)
            assert mirror.board_fen() == board.board_fen(), board.fen()
            assert mirror.turn == board.turn
            assert mirror.castling_rights == board.clean_castling_rights()
            assert mirror.halfmove_clock == board.halfmove_clock
            assert stacks.keys[ply] == bb.compute_key(stacks.pieces, stacks.state, ply)


def test_null_move_flips_side_and_clears_en_passant() -> None:
    stacks = _stacks_for("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    bb.make_null(stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, 0)
    assert stacks.state[1, bb.STM] == bb.BLACK
    assert stacks.state[1, bb.EP] == -1
    assert stacks.keys[1] == bb.compute_key(stacks.pieces, stacks.state, 1)
    assert stacks.keys[1] != stacks.keys[0]
```

- [ ] **Step 2: Run to verify failure**: the new tests fail with `AttributeError: module 'bitboard' has no attribute 'new_stacks'`.

- [ ] **Step 3: Append to `bitboard.py`**

```python
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
    return key


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
        if rights & (kingside | queenside) and not is_attacked(
            pieces[ply], king_sq, stm ^ 1, occ
        ):
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
    return ZOBRIST_PIECE[code, sq]


@_jit
def _put_piece(
    pieces: npt.NDArray[np.uint64], mailbox: npt.NDArray[np.int8], ply: int, sq: int, code: int
) -> np.uint64:
    pieces[ply, code] |= bit(sq)
    mailbox[ply, sq] = code
    return ZOBRIST_PIECE[code, sq]


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
```

- [ ] **Step 4: Run**: `uv run pytest tests/test_bitboard.py -q`. The perft numbers are the Chess Programming Wiki's; a mismatch at depth 1 means generation, deeper usually en passant, rights after a rook capture, or promotion captures. The differential test prints the first disagreeing FEN.

- [ ] **Step 5: Lint, types, commit**

```bash
uv run ruff check . && uv run mypy
git add bitboard.py tests/test_bitboard.py
git commit -m "feat(bitboard): move generation, copy-make, zobrist keys and perft"
```

---

### Task 3: NNUE accumulators on the bitboard arrays

**Files:**
- Create: `nnue_bitboard.py`
- Test: `tests/test_nnue_bitboard.py`

**Interfaces:**
- Consumes: `bitboard` constants, `BoardStacks`, `make_move`, `generate_moves`, `piece_code`.
- Produces: `AccStacks` NamedTuple (`white_acc int16[MAX_PLY, L1]`, `black_acc`, `white_psqt int32[MAX_PLY, 8]`, `black_psqt`, `act int64[L1]`, `l1c int64[L2]`, `l1x int64[2*L2]`, `l2c int64[L3]`, `l2x int64[2*L3]`), `new_acc_stacks()`, kernels `feature_index(pov, king_sq, sq, code) -> int`, `refresh_ply(ft_w, ft_b, psqt_w, mailbox, state, ply, white_acc, black_acc, white_psqt, black_psqt)`, `update_ply(ft_w, ft_b, psqt_w, mailbox, state, ply, move, white_acc, black_acc, white_psqt, black_psqt)` (reads ply, writes ply + 1; call after a successful `make_move`), `copy_ply(ply, white_acc, black_acc, white_psqt, black_psqt)` (null move), `evaluate(ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt, act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b) -> int`.

- [ ] **Step 1: Write the failing tests** (`tests/test_nnue_bitboard.py`)

```python
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
            ply, stacks.state, stacks.occupied,
            acc.white_acc, acc.black_acc, acc.white_psqt, acc.black_psqt,
            acc.act, acc.l1c, acc.l1x, acc.l2c, acc.l2x,
            _NET.l1_w, _NET.l1_b, _NET.l2_w, _NET.l2_b, _NET.out_w, _NET.out_b,
        )
    )


def refresh(stacks: bb.BoardStacks, acc: nb.AccStacks, ply: int) -> None:
    nb.refresh_ply(
        _NET.ft_w, _NET.ft_b, _NET.psqt_w, stacks.mailbox, stacks.state, ply,
        acc.white_acc, acc.black_acc, acc.white_psqt, acc.black_psqt,
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
        ply = 0
        for _ in range(80):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            count = bb.generate_moves(
                stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, ply, moves, False
            )
            ours = [int(m) for m in moves[ply, :count] if bb.move_to_uci(int(m)) == move.uci()]
            assert bb.make_move(
                stacks.pieces, stacks.occupied, stacks.mailbox, stacks.state, stacks.keys, ply,
                ours[0],
            )
            nb.update_ply(
                _NET.ft_w, _NET.ft_b, _NET.psqt_w, stacks.mailbox, stacks.state, ply, ours[0],
                acc.white_acc, acc.black_acc, acc.white_psqt, acc.black_psqt,
            )
            engine.push(move)
            board.push(move)
            ply += 1
            assert evaluate(stacks, acc, ply) == engine.evaluate(), board.fen()
```

- [ ] **Step 2: Run to verify failure**: `ModuleNotFoundError: No module named 'nnue_bitboard'`.

- [ ] **Step 3: Write `nnue_bitboard.py`**

```python
"""HalfKAv2_hm features and accumulators over the bitboard stacks.

The index formula is nnue_features.feature_index on piece codes instead of python-chess
pieces; tests pin the two against each other. Accumulators keep the shapes and dtypes
of nnue_engine so tests/reference.py still pins the arithmetic, and _forward is the
same kernel moved here. A perspective whose own king moved is refreshed, as before.
"""

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from numba import njit

from bitboard import (
    BKING,
    BLACK,
    EMPTY,
    FLAG_CASTLE,
    FLAG_EP,
    KING,
    MAX_PLY,
    ROOK,
    STM,
    WHITE,
    WKING,
    popcount,
)
from nnue_arch import (
    FT_ACT_MAX,
    FT_PAIRWISE_SHIFT,
    HIDDEN_ACT_MAX,
    L1,
    L1_LINEAR_SHIFT,
    L1_SQUARE_SHIFT,
    L2,
    L2_LINEAR_SHIFT,
    L2_SQUARE_SHIFT,
    L3,
    NUM_PLANES,
    NUM_PSQT_BUCKETS,
    NUM_SQ,
    OUTPUT_DIV,
    OUTPUT_MUL,
)
from nnue_features import KING_BUCKETS

KING_BUCKET_TABLE = np.asarray(KING_BUCKETS, dtype=np.int64)


def _jit[F: Callable[..., Any]](function: F) -> F:
    return cast("F", njit(cache=False)(function))


class AccStacks(NamedTuple):
    white_acc: npt.NDArray[np.int16]
    black_acc: npt.NDArray[np.int16]
    white_psqt: npt.NDArray[np.int32]
    black_psqt: npt.NDArray[np.int32]
    act: npt.NDArray[np.int64]
    l1c: npt.NDArray[np.int64]
    l1x: npt.NDArray[np.int64]
    l2c: npt.NDArray[np.int64]
    l2x: npt.NDArray[np.int64]


def new_acc_stacks() -> AccStacks:
    return AccStacks(
        np.zeros((MAX_PLY, L1), dtype=np.int16),
        np.zeros((MAX_PLY, L1), dtype=np.int16),
        np.zeros((MAX_PLY, NUM_PSQT_BUCKETS), dtype=np.int32),
        np.zeros((MAX_PLY, NUM_PSQT_BUCKETS), dtype=np.int32),
        np.zeros(L1, dtype=np.int64),
        np.zeros(L2, dtype=np.int64),
        np.zeros(2 * L2, dtype=np.int64),
        np.zeros(L3, dtype=np.int64),
        np.zeros(2 * L3, dtype=np.int64),
    )


@_jit
def feature_index(pov: int, king_sq: int, sq: int, code: int) -> int:
    """Feature row for (square, piece code) from perspective `pov` (0 white, 1 black)."""
    horizontal = 7 if (king_sq & 7) < 4 else 0
    vertical = 0 if pov == WHITE else 56
    bucket = KING_BUCKET_TABLE[king_sq ^ horizontal ^ vertical]
    own = 0 if (code // 6) == pov else 1
    plane = (code % 6) * 2 + own
    return (sq ^ horizontal ^ vertical) + NUM_SQ * plane + NUM_PLANES * bucket


@_jit
def _add_row(
    ft_w: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    row: int,
    acc: npt.NDArray[np.int16],
    psqt: npt.NDArray[np.int32],
    ply: int,
) -> None:
    for j in range(L1):
        acc[ply, j] += ft_w[row, j]
    for j in range(NUM_PSQT_BUCKETS):
        psqt[ply, j] += psqt_w[row, j]


@_jit
def _sub_row(
    ft_w: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    row: int,
    acc: npt.NDArray[np.int16],
    psqt: npt.NDArray[np.int32],
    ply: int,
) -> None:
    for j in range(L1):
        acc[ply, j] -= ft_w[row, j]
    for j in range(NUM_PSQT_BUCKETS):
        psqt[ply, j] -= psqt_w[row, j]


@_jit
def _refresh_one(
    ft_w: npt.NDArray[np.int16],
    ft_b: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    mailbox: npt.NDArray[np.int8],
    ply: int,
    pov: int,
    king_sq: int,
    acc: npt.NDArray[np.int16],
    psqt: npt.NDArray[np.int32],
) -> None:
    for j in range(L1):
        acc[ply, j] = ft_b[j]
    for j in range(NUM_PSQT_BUCKETS):
        psqt[ply, j] = 0
    for sq in range(64):
        code = mailbox[ply, sq]
        if code != EMPTY:
            _add_row(ft_w, psqt_w, feature_index(pov, king_sq, sq, code), acc, psqt, ply)


@_jit
def refresh_ply(
    ft_w: npt.NDArray[np.int16],
    ft_b: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    ply: int,
    white_acc: npt.NDArray[np.int16],
    black_acc: npt.NDArray[np.int16],
    white_psqt: npt.NDArray[np.int32],
    black_psqt: npt.NDArray[np.int32],
) -> None:
    _refresh_one(ft_w, ft_b, psqt_w, mailbox, ply, WHITE, state[ply, WKING], white_acc, white_psqt)
    _refresh_one(ft_w, ft_b, psqt_w, mailbox, ply, BLACK, state[ply, BKING], black_acc, black_psqt)


@_jit
def _update_one(
    ft_w: npt.NDArray[np.int16],
    ft_b: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    ply: int,
    move: int,
    pov: int,
    acc: npt.NDArray[np.int16],
    psqt: npt.NDArray[np.int32],
) -> None:
    """acc[ply + 1] from acc[ply] for one perspective.

    mailbox[ply] is the position before the move and mailbox[ply + 1] after it, so this
    runs after make_move.
    """
    nxt = ply + 1
    frm = move & 63
    to = (move >> 6) & 63
    flags = move >> 15
    mover = mailbox[ply, frm]
    mover_colour = mover // 6
    if mover % 6 == KING and mover_colour == pov:
        _refresh_one(ft_w, ft_b, psqt_w, mailbox, nxt, pov, state[nxt, WKING + pov], acc, psqt)
        return
    king_sq = state[ply, WKING + pov]
    for j in range(L1):
        acc[nxt, j] = acc[ply, j]
    for j in range(NUM_PSQT_BUCKETS):
        psqt[nxt, j] = psqt[ply, j]
    _sub_row(ft_w, psqt_w, feature_index(pov, king_sq, frm, mover), acc, psqt, nxt)
    _add_row(ft_w, psqt_w, feature_index(pov, king_sq, to, mailbox[nxt, to]), acc, psqt, nxt)
    if flags & FLAG_EP:
        captured_sq = to - 8 if mover_colour == WHITE else to + 8
        captured = mailbox[ply, captured_sq]
        _sub_row(ft_w, psqt_w, feature_index(pov, king_sq, captured_sq, captured), acc, psqt, nxt)
    elif mailbox[ply, to] != EMPTY:
        _sub_row(ft_w, psqt_w, feature_index(pov, king_sq, to, mailbox[ply, to]), acc, psqt, nxt)
    if flags & FLAG_CASTLE:
        rook = ROOK + mover_colour * 6
        if to > frm:
            rook_from, rook_to = frm + 3, frm + 1
        else:
            rook_from, rook_to = frm - 4, frm - 1
        _sub_row(ft_w, psqt_w, feature_index(pov, king_sq, rook_from, rook), acc, psqt, nxt)
        _add_row(ft_w, psqt_w, feature_index(pov, king_sq, rook_to, rook), acc, psqt, nxt)


@_jit
def update_ply(
    ft_w: npt.NDArray[np.int16],
    ft_b: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    mailbox: npt.NDArray[np.int8],
    state: npt.NDArray[np.int32],
    ply: int,
    move: int,
    white_acc: npt.NDArray[np.int16],
    black_acc: npt.NDArray[np.int16],
    white_psqt: npt.NDArray[np.int32],
    black_psqt: npt.NDArray[np.int32],
) -> None:
    _update_one(ft_w, ft_b, psqt_w, mailbox, state, ply, move, WHITE, white_acc, white_psqt)
    _update_one(ft_w, ft_b, psqt_w, mailbox, state, ply, move, BLACK, black_acc, black_psqt)


@_jit
def copy_ply(
    ply: int,
    white_acc: npt.NDArray[np.int16],
    black_acc: npt.NDArray[np.int16],
    white_psqt: npt.NDArray[np.int32],
    black_psqt: npt.NDArray[np.int32],
) -> None:
    """Carry ply's accumulators to ply + 1 unchanged (a null move)."""
    nxt = ply + 1
    for j in range(L1):
        white_acc[nxt, j] = white_acc[ply, j]
        black_acc[nxt, j] = black_acc[ply, j]
    for j in range(NUM_PSQT_BUCKETS):
        white_psqt[nxt, j] = white_psqt[ply, j]
        black_psqt[nxt, j] = black_psqt[ply, j]


@_jit
def _forward(
    stm_acc: npt.NDArray[np.int16],
    ntm_acc: npt.NDArray[np.int16],
    psqt_diff: int,
    white_to_move: bool,
    l1_w: npt.NDArray[np.int8],
    l1_b: npt.NDArray[np.int32],
    l2_w: npt.NDArray[np.int8],
    l2_b: npt.NDArray[np.int32],
    out_w: npt.NDArray[np.int8],
    out_b: int,
    act: npt.NDArray[np.int64],
    l1c: npt.NDArray[np.int64],
    l1x: npt.NDArray[np.int64],
    l2c: npt.NDArray[np.int64],
    l2x: npt.NDArray[np.int64],
) -> int:
    # Body copied verbatim from nnue_engine._forward (lines 98-157 at 74a7d63).
    ...


@_jit
def evaluate(
    ply: int,
    state: npt.NDArray[np.int32],
    occupied: npt.NDArray[np.uint64],
    white_acc: npt.NDArray[np.int16],
    black_acc: npt.NDArray[np.int16],
    white_psqt: npt.NDArray[np.int32],
    black_psqt: npt.NDArray[np.int32],
    act: npt.NDArray[np.int64],
    l1c: npt.NDArray[np.int64],
    l1x: npt.NDArray[np.int64],
    l2c: npt.NDArray[np.int64],
    l2x: npt.NDArray[np.int64],
    l1_w: npt.NDArray[np.int8],
    l1_b: npt.NDArray[np.int32],
    l2_w: npt.NDArray[np.int8],
    l2_b: npt.NDArray[np.int32],
    out_w: npt.NDArray[np.int8],
    out_b: npt.NDArray[np.int32],
) -> int:
    """Score in 1/32 cp for the side to move at `ply`."""
    bucket = (popcount(occupied[ply, 2]) - 1) // 4
    psqt_diff = int(white_psqt[ply, bucket]) - int(black_psqt[ply, bucket])
    if state[ply, STM] == WHITE:
        return _forward(
            white_acc[ply], black_acc[ply], psqt_diff, True, l1_w[bucket], l1_b[bucket],
            l2_w[bucket], l2_b[bucket], out_w[bucket], int(out_b[bucket]),
            act, l1c, l1x, l2c, l2x,
        )
    return _forward(
        black_acc[ply], white_acc[ply], psqt_diff, False, l1_w[bucket], l1_b[bucket],
        l2_w[bucket], l2_b[bucket], out_w[bucket], int(out_b[bucket]),
        act, l1c, l1x, l2c, l2x,
    )
```

The `...` in `_forward` stands for the verbatim body of `nnue_engine._forward` and nothing else.

- [ ] **Step 4: Run**: `uv run pytest tests/test_nnue_bitboard.py tests/test_parity.py -q`. A mismatch in the incremental test only points at `_update_one` (captured piece read from the wrong ply, or the castling rook).

- [ ] **Step 5: Lint, types, commit** (add `"nnue_bitboard.py"` to mypy files):

```bash
uv run ruff check . && uv run mypy
git add nnue_bitboard.py tests/test_nnue_bitboard.py pyproject.toml
git commit -m "feat(nnue): accumulators and evaluation over the bitboard stacks"
```

---

### Task 4: The search kernel and the new `Searcher`

**Files:**
- Create: `search_kernel.py`
- Rewrite: `search.py`
- Rewrite: `tests/test_search.py`
- Modify: `agent.py` (imports only)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `search_kernel.search(depth, ply, alpha, beta, allow_null, board, acc, net, tables, ctrl, deadline) -> int` with `board = tuple(BoardStacks)`, `acc = tuple(AccStacks)`, `net = (ft_w, ft_b, psqt_w, l1_w, l1_b, l2_w, l2_b, out_w, out_b)`, `tables = (tt_keys uint64[TT_SIZE], tt_data int64[TT_SIZE], killers int32[MAX_PLY_LIMIT+2, 2], history int64[64, 64], moves int32[MAX_PLY, MAX_MOVES], scores int64[MAX_PLY, MAX_MOVES], game_keys uint64[1024], see_gain int64[32])`, `ctrl int64[8]` = `[nodes, abort, root_best_move, node_limit, root_hint, generation, game_key_count, unused]`; `search.Searcher(net)` with `pick(fen, time_left_ms, node_limit=0) -> chess.Move`, `warm_up()`, `note_root_position(board)`, `is_draw_key(key)`, `nodes`; module functions `budget_ms`, `aspiration_window`, `widen`, `to_tt_score`, `from_tt_score`, `load_searcher(path=None)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_search.py`, replacing the file)

```python
"""Behavioural tests for the bitboard search.

Strength is judged by SPRT, not here; these pin the rules: legality, mates, draws, the
clock, and the table's bookkeeping.
"""

import random
import time

import chess
import pytest

import bitboard as bb
import search
import search_kernel as sk
from search import Searcher
from tools.gen_random_net import random_network

_NET = random_network(2026)


@pytest.fixture(scope="module")
def searcher() -> Searcher:
    s = Searcher(_NET)
    s.warm_up()
    return s


def test_budget_scales_with_clock() -> None:
    assert search.budget_ms(120_000) > search.budget_ms(12_000)


def test_budget_never_exceeds_fraction_of_clock() -> None:
    assert search.budget_ms(100) <= search.MAX_FRACTION * 100


def test_budget_stays_positive_on_a_nearly_dead_clock() -> None:
    assert search.budget_ms(1) >= search.MIN_BUDGET_MS


def test_mate_scores_are_stored_relative_to_the_node() -> None:
    assert search.to_tt_score(sk.MATE - 5, 3) == sk.MATE - 2
    assert search.from_tt_score(sk.MATE - 2, 3) == sk.MATE - 5
    assert search.to_tt_score(-sk.MATE + 5, 3) == -sk.MATE + 2
    assert search.to_tt_score(1234, 3) == 1234


def test_tt_packing_round_trips() -> None:
    data = sk.pack_tt(12, sk.LOWER, 9, bb.encode_move(12, 28, 0, bb.FLAG_DOUBLE), -123456)
    assert sk.tt_depth(data) == 12
    assert sk.tt_bound(data) == sk.LOWER
    assert sk.tt_generation(data) == 9
    assert sk.tt_move(data) == bb.encode_move(12, 28, 0, bb.FLAG_DOUBLE)
    assert sk.tt_score(data) == -123456


def test_aspiration_window_brackets_the_previous_score() -> None:
    alpha, beta, window = search.aspiration_window(1000, search.ASPIRATION_MIN_DEPTH)
    assert alpha == 1000 - window
    assert beta == 1000 + window
    assert search.aspiration_window(1000, 1)[0] == -2 * sk.MATE


def test_widen_opens_only_the_failed_side() -> None:
    alpha, beta, window = search.widen(900, 1100, 850, 100)
    assert (alpha, beta, window) == (850 - 200, 1100, 200)


def test_see_scores_a_free_pawn_and_a_defended_one(searcher: Searcher) -> None:
    def see_of(fen: str, uci: str) -> int:
        board = chess.Board(fen)
        bb.set_from_board(board, searcher.board, 0)
        move = chess.Move.from_uci(uci)
        flags = bb.FLAG_CAPTURE if board.is_capture(move) else 0
        encoded = bb.encode_move(move.from_square, move.to_square, 0, flags)
        return int(
            sk.see(
                searcher.board.pieces[0], searcher.board.mailbox[0],
                int(searcher.board.state[0, bb.STM]), searcher.board.occupied[0, 2],
                encoded, searcher.see_gain,
            )
        )

    assert see_of("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 1
    assert see_of("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 0
    assert see_of("4k3/8/2p5/3p4/8/8/8/4K2R w - - 0 1", "h1h7") == 0
    assert see_of("4k3/8/2p5/3p4/4N3/8/8/4K3 w - - 0 1", "e4d5") == -2


def test_finds_mate_in_one(searcher: Searcher) -> None:
    assert searcher.pick("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 2000).uci() == "a1a8"


def test_finds_mate_in_two_through_a_quiet_first_move(searcher: Searcher) -> None:
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    assert searcher.pick(fen, 3000).uci() == "h5f7"


def test_takes_the_free_queen(searcher: Searcher) -> None:
    fen = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    assert searcher.pick(fen, 1500).uci() == "f3h4"


def test_returns_a_legal_move_when_in_check(searcher: Searcher) -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/7P/5P2/PPPPP1Pq/RNBQKBNR w KQkq - 1 3")
    assert searcher.pick(board.fen(), 1000) in board.legal_moves


def test_never_returns_an_illegal_move(searcher: Searcher) -> None:
    rng = random.Random(8)
    for _game in range(40):
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over():
                break
            if rng.random() < 0.3:
                move = searcher.pick(board.fen(), 300)
                assert move in board.legal_moves, board.fen()
            else:
                move = rng.choice(list(board.legal_moves))
            board.push(move)


def test_respects_a_tight_time_budget(searcher: Searcher) -> None:
    start = time.monotonic()
    searcher.pick("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 1000)
    assert time.monotonic() - start < 0.5


def test_node_limit_makes_the_search_deterministic(searcher: Searcher) -> None:
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    first = searcher.pick(fen, 60_000, node_limit=20_000)
    searcher.warm_up()
    second = searcher.pick(fen, 60_000, node_limit=20_000)
    assert first == second


def test_no_legal_moves_raises(searcher: Searcher) -> None:
    with pytest.raises(ValueError):
        searcher.pick("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1000)


def test_root_position_is_remembered_for_repetition(searcher: Searcher) -> None:
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    searcher.note_root_position(board)
    stacks = bb.new_stacks()
    bb.set_from_board(board, stacks, 0)
    assert searcher.is_draw_key(stacks.keys[0])


def test_avoids_a_threefold_when_winning(searcher: Searcher) -> None:
    # K+Q vs K: after seeing the position twice, the search must not shuffle into it again.
    searcher.warm_up()
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    seen: dict[str, int] = {}
    for _ in range(30):
        if board.is_game_over():
            break
        board.push(searcher.pick(board.fen(), 1000))
        if board.is_game_over():
            break
        board.push(min(board.legal_moves, key=lambda m: m.uci()))
        key = board.board_fen()
        seen[key] = seen.get(key, 0) + 1
    assert board.is_checkmate() or max(seen.values()) < 3


def test_agent_returns_legal_uci() -> None:
    import agent

    board = chess.Board()
    assert chess.Move.from_uci(agent.get_move(board.fen(), 5000)) in board.legal_moves


def test_agent_falls_back_rather_than_raising_on_a_dead_clock() -> None:
    import agent

    move = chess.Move.from_uci(agent.get_move(chess.STARTING_FEN, 0))
    assert move in chess.Board().legal_moves
```

- [ ] **Step 2: Run to verify failure**: `ModuleNotFoundError: No module named 'search_kernel'`.

- [ ] **Step 3: Write `search_kernel.py`**

```python
"""The alpha-beta search as one recursive numba kernel over the bitboard stacks.

Quiescence is the depth <= 0 branch of the same function, so the only recursion is
self-recursion, which numba resolves from the base cases. Every constant and rule is
the python-chess search's at 74a7d63; the kernel exists to run them ten times faster,
and the first gate measures exactly that.

Scores are integers in 1/32 cp for the side to move. A mate at distance `ply` scores
MATE - ply; the table stores mates relative to the node and converts on probe.

ctrl layout: [nodes, abort, root best move, node limit (0 = none), root hint move,
generation, game key count, unused]. The kernel never raises: the clock sets ctrl[1]
and every level unwinds on it.
"""

import time
from collections.abc import Callable
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from numba import njit, objmode

from bitboard import (
    BISHOP,
    EMPTY,
    FLAG_CAPTURE,
    FLAG_EP,
    HALFMOVE,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    STM,
    WHITE,
    WKING,
    ZERO,
    attackers_to,
    bit,
    generate_moves,
    is_attacked,
    lsb,
    make_move,
    make_null,
    popcount,
)
from nnue_arch import SCORE_PER_CP
from nnue_bitboard import copy_ply, evaluate, update_ply

MATE: Final = 1_000_000
MAX_DEPTH: Final = 64
MAX_PLY_LIMIT: Final = 128
MATE_THRESHOLD: Final = MATE - MAX_PLY_LIMIT

TT_BITS: Final = 20
TT_SIZE: Final = 1 << TT_BITS
CLOCK_CHECK_NODES: Final = 4096

# Ordering scores; only their order matters.
TT_MOVE_BONUS: Final = 1_000_000
CAPTURE_BONUS: Final = 500_000
KILLER_BONUS: Final = 400_000
LOSING_CAPTURE_BONUS: Final = 300_000

NULL_MIN_DEPTH: Final = 3
NULL_REDUCTION: Final = 2
LMR_MIN_DEPTH: Final = 3
LMR_MIN_MOVES: Final = 4
LMR_LATE_MOVES: Final = 8
LMR_DEEP: Final = 6
RFP_MAX_DEPTH: Final = 3
RFP_MARGIN: Final = 120 * SCORE_PER_CP
DELTA_MARGIN: Final = 400 * SCORE_PER_CP

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

# Indexed by piece type. MVV-LVA values in pawns; delta-pruning victim values in the
# net's units (the phase 5 spec explains why a knight is ~1700 there).
PIECE_VALUE: Final = np.array([1, 3, 3, 5, 9, 20], dtype=np.int64)
VICTIM_VALUE: Final = np.array([200, 1700, 1700, 1900, 3300, 0], dtype=np.int64) * SCORE_PER_CP

CTRL_NODES: Final = 0
CTRL_ABORT: Final = 1
CTRL_ROOT_MOVE: Final = 2
CTRL_NODE_LIMIT: Final = 3
CTRL_ROOT_HINT: Final = 4
CTRL_GENERATION: Final = 5
CTRL_GAME_KEYS: Final = 6


def _jit[F: Callable[..., Any]](function: F) -> F:
    return cast("F", njit(cache=False)(function))


@_jit
def now() -> float:
    with objmode(t="float64"):
        t = time.monotonic()
    return t


@_jit
def to_tt_score(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


@_jit
def from_tt_score(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


@_jit
def pack_tt(depth: int, bound: int, generation: int, move: int, score: int) -> int:
    low = (move & 0x7FFFF) | ((depth & 0x7F) << 19) | ((bound & 3) << 26)
    low |= (generation & 15) << 28
    return (score << 32) | low


@_jit
def tt_move(data: int) -> int:
    return data & 0x7FFFF


@_jit
def tt_depth(data: int) -> int:
    return (data >> 19) & 0x7F


@_jit
def tt_bound(data: int) -> int:
    return (data >> 26) & 3


@_jit
def tt_generation(data: int) -> int:
    return (data >> 28) & 15


@_jit
def tt_score(data: int) -> int:
    return data >> 32


@_jit
def is_tactical(move: int) -> bool:
    return (move >> 15) & FLAG_CAPTURE != 0 or (move >> 12) & 7 != 0


@_jit
def victim_type(mailbox_row: npt.NDArray[np.int8], move: int) -> int:
    if (move >> 15) & FLAG_EP:
        return PAWN
    code = mailbox_row[(move >> 6) & 63]
    return PAWN if code == EMPTY else code % 6


@_jit
def _least_valuable(pieces_row: npt.NDArray[np.uint64], attackers: np.uint64, colour: int) -> int:
    """Piece type of the cheapest attacker of `colour` within `attackers`, or -1."""
    for piece in range(6):
        if attackers & pieces_row[colour * 6 + piece] != ZERO:
            return piece
    return -1


@_jit
def see(
    pieces_row: npt.NDArray[np.uint64],
    mailbox_row: npt.NDArray[np.int8],
    stm: int,
    occ: np.uint64,
    move: int,
    gain: npt.NDArray[np.int64],
) -> int:
    """Static exchange evaluation in pawn units: the phase 5 swap list on bitboards."""
    frm = move & 63
    to = (move >> 6) & 63
    flags = move >> 15
    piece = mailbox_row[frm] % 6
    gain[0] = PIECE_VALUE[victim_type(mailbox_row, move)] if flags & FLAG_CAPTURE else 0
    occ &= ~bit(frm)
    if flags & FLAG_EP:
        occ &= ~bit(to - 8 if stm == WHITE else to + 8)
    side = stm ^ 1
    depth = 1
    while True:
        gain[depth] = PIECE_VALUE[piece] - gain[depth - 1]
        attackers = attackers_to(pieces_row, to, occ) & occ
        piece = _least_valuable(pieces_row, attackers, side)
        if piece < 0 or depth >= 30:
            break
        candidates = attackers & pieces_row[side * 6 + piece]
        occ &= ~bit(lsb(candidates))
        side ^= 1
        depth += 1
    depth -= 1  # the last entry assumed a recapture that never came
    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return gain[0]


@_jit
def has_major_material(pieces_row: npt.NDArray[np.uint64], stm: int) -> bool:
    base = stm * 6
    majors = pieces_row[base + KNIGHT] | pieces_row[base + BISHOP]
    majors |= pieces_row[base + ROOK] | pieces_row[base + QUEEN]
    return majors != ZERO


@_jit
def insufficient_material(pieces_row: npt.NDArray[np.uint64]) -> bool:
    heavy = pieces_row[PAWN] | pieces_row[6 + PAWN] | pieces_row[ROOK] | pieces_row[6 + ROOK]
    heavy |= pieces_row[QUEEN] | pieces_row[6 + QUEEN]
    if heavy != ZERO:
        return False
    minors = pieces_row[KNIGHT] | pieces_row[BISHOP] | pieces_row[6 + KNIGHT]
    minors |= pieces_row[6 + BISHOP]
    return popcount(minors) <= 1


@_jit
def is_repetition(
    keys: npt.NDArray[np.uint64], ply: int, game_keys: npt.NDArray[np.uint64], n_game_keys: int
) -> bool:
    key = keys[ply]
    for i in range(ply):
        if keys[i] == key:
            return True
    for i in range(n_game_keys):
        if game_keys[i] == key:
            return True
    return False


@_jit
def score_moves(
    mailbox_row: npt.NDArray[np.int8],
    pieces_row: npt.NDArray[np.uint64],
    stm: int,
    occ: np.uint64,
    moves_row: npt.NDArray[np.int32],
    scores_row: npt.NDArray[np.int64],
    count: int,
    hash_move: int,
    killers_row: npt.NDArray[np.int32],
    history: npt.NDArray[np.int64],
    gain: npt.NDArray[np.int64],
) -> None:
    for i in range(count):
        move = moves_row[i]
        if move == hash_move:
            scores_row[i] = TT_MOVE_BONUS
        elif is_tactical(move):
            victim = victim_type(mailbox_row, move)
            attacker = mailbox_row[move & 63] % 6
            value = 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
            promotion = (move >> 12) & 7
            if promotion:
                value += 10 * PIECE_VALUE[promotion]
            if promotion == 0 and see(pieces_row, mailbox_row, stm, occ, move, gain) < 0:
                scores_row[i] = LOSING_CAPTURE_BONUS + value
            else:
                scores_row[i] = CAPTURE_BONUS + value
        elif move == killers_row[0] or move == killers_row[1]:
            scores_row[i] = KILLER_BONUS
        else:
            scores_row[i] = history[move & 63, (move >> 6) & 63]


@_jit
def pick_next(
    moves_row: npt.NDArray[np.int32], scores_row: npt.NDArray[np.int64], start: int, count: int
) -> int:
    """Swap the best-scored remaining move into `start` and return it."""
    best = start
    for i in range(start + 1, count):
        if scores_row[i] > scores_row[best]:
            best = i
    if best != start:
        moves_row[start], moves_row[best] = moves_row[best], moves_row[start]
        scores_row[start], scores_row[best] = scores_row[best], scores_row[start]
    return moves_row[start]


@_jit
def search(  # noqa: PLR0912, PLR0915
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
    allow_null: bool,
    board: tuple[Any, ...],
    acc: tuple[Any, ...],
    net: tuple[Any, ...],
    tables: tuple[Any, ...],
    ctrl: npt.NDArray[np.int64],
    deadline: float,
) -> int:
    pieces, occupied, mailbox, state, keys = board
    white_acc, black_acc, white_psqt, black_psqt, act, l1c, l1x, l2c, l2x = acc
    ft_w, ft_b, psqt_w, l1_w, l1_b, l2_w, l2_b, out_w, out_b = net
    tt_keys, tt_data, killers, history, moves, scores, game_keys, gain = tables

    ctrl[CTRL_NODES] += 1
    if ctrl[CTRL_NODES] % CLOCK_CHECK_NODES == 0:
        limit = ctrl[CTRL_NODE_LIMIT]
        if now() > deadline or (limit > 0 and ctrl[CTRL_NODES] >= limit):
            ctrl[CTRL_ABORT] = 1
    if ctrl[CTRL_ABORT]:
        return 0

    stm = state[ply, STM]
    occ = occupied[ply, 2]
    key = keys[ply]
    pieces_row = pieces[ply]
    mailbox_row = mailbox[ply]

    if ply > 0 and (
        state[ply, HALFMOVE] >= 100
        or insufficient_material(pieces_row)
        or is_repetition(keys, ply, game_keys, ctrl[CTRL_GAME_KEYS])
    ):
        return 0

    hash_move = -1
    slot = int(key & np.uint64(TT_SIZE - 1))
    if tt_keys[slot] == key:
        data = tt_data[slot]
        hash_move = tt_move(data)
        if ply > 0 and tt_depth(data) >= depth:
            stored = from_tt_score(tt_score(data), ply)
            bound = tt_bound(data)
            if bound == EXACT:
                return stored
            if bound == LOWER and stored >= beta:
                return stored
            if bound == UPPER and stored <= alpha:
                return stored
    if ply == 0 and ctrl[CTRL_ROOT_HINT] >= 0:
        hash_move = ctrl[CTRL_ROOT_HINT]

    in_check = is_attacked(pieces_row, state[ply, WKING + stm], stm ^ 1, occ)

    # ---- quiescence ----------------------------------------------------------------------
    if depth <= 0 or ply >= MAX_PLY_LIMIT:
        if ply >= MAX_PLY_LIMIT:
            return evaluate(
                ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
                act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
            )
        static = -2 * MATE
        best = -2 * MATE
        if in_check:
            count = generate_moves(pieces, occupied, mailbox, state, ply, moves, False)
        else:
            static = evaluate(
                ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
                act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
            )
            if static >= beta:
                return static
            if static > alpha:
                alpha = static
            best = static
            count = generate_moves(pieces, occupied, mailbox, state, ply, moves, True)
        score_moves(
            mailbox_row, pieces_row, stm, occ, moves[ply], scores[ply], count, -1,
            killers[ply], history, gain,
        )
        legal = 0
        for i in range(count):
            move = pick_next(moves[ply], scores[ply], i, count)
            if not in_check and (move >> 12) & 7 == 0:
                victim = victim_type(mailbox_row, move)
                if static + VICTIM_VALUE[victim] + DELTA_MARGIN < alpha:
                    continue
                if see(pieces_row, mailbox_row, stm, occ, move, gain) < 0:
                    continue
            if not make_move(pieces, occupied, mailbox, state, keys, ply, move):
                continue
            legal += 1
            update_ply(
                ft_w, ft_b, psqt_w, mailbox, state, ply, move,
                white_acc, black_acc, white_psqt, black_psqt,
            )
            score = -search(
                depth - 1, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
            )
            if ctrl[CTRL_ABORT]:
                return 0
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if in_check and legal == 0:
            return -MATE + ply
        return best

    # ---- main search ---------------------------------------------------------------------
    if ply > 0 and depth <= RFP_MAX_DEPTH and not in_check and beta < MATE_THRESHOLD:
        static = evaluate(
            ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
            act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
        )
        if static - RFP_MARGIN * depth >= beta:
            return static

    if (
        ply > 0
        and allow_null
        and depth >= NULL_MIN_DEPTH
        and not in_check
        and beta < MATE_THRESHOLD
        and has_major_material(pieces_row, stm)
    ):
        make_null(pieces, occupied, mailbox, state, keys, ply)
        copy_ply(ply, white_acc, black_acc, white_psqt, black_psqt)
        passed = -search(
            depth - 1 - NULL_REDUCTION, ply + 1, -beta, -beta + 1, False,
            board, acc, net, tables, ctrl, deadline,
        )
        if ctrl[CTRL_ABORT]:
            return 0
        if passed >= beta:
            return beta

    count = generate_moves(pieces, occupied, mailbox, state, ply, moves, False)
    score_moves(
        mailbox_row, pieces_row, stm, occ, moves[ply], scores[ply], count, hash_move,
        killers[ply], history, gain,
    )

    original_alpha = alpha
    best = -2 * MATE
    best_move = -1
    legal = 0
    for i in range(count):
        move = pick_next(moves[ply], scores[ply], i, count)
        if not make_move(pieces, occupied, mailbox, state, keys, ply, move):
            continue
        update_ply(
            ft_w, ft_b, psqt_w, mailbox, state, ply, move,
            white_acc, black_acc, white_psqt, black_psqt,
        )
        index = legal
        legal += 1
        tactical = is_tactical(move)
        gives_check = is_attacked(
            pieces[ply + 1], state[ply + 1, WKING + (stm ^ 1)], stm, occupied[ply + 1, 2]
        )
        child = depth - 1 + (1 if gives_check else 0)
        reduction = 0
        if (
            depth >= LMR_MIN_DEPTH
            and index >= LMR_MIN_MOVES
            and not tactical
            and not in_check
            and not gives_check
        ):
            reduction = 2 if (index >= LMR_LATE_MOVES and depth >= LMR_DEEP) else 1
        if index == 0:
            score = -search(
                child, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
            )
        else:
            score = -search(
                child - reduction, ply + 1, -alpha - 1, -alpha, True,
                board, acc, net, tables, ctrl, deadline,
            )
            if reduction and score > alpha and not ctrl[CTRL_ABORT]:
                score = -search(
                    child, ply + 1, -alpha - 1, -alpha, True,
                    board, acc, net, tables, ctrl, deadline,
                )
            if alpha < score < beta and not ctrl[CTRL_ABORT]:
                score = -search(
                    child, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
                )
        if ctrl[CTRL_ABORT]:
            return 0
        if score > best:
            best = score
            best_move = move
            if ply == 0:
                ctrl[CTRL_ROOT_MOVE] = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if not tactical:
                if killers[ply, 0] != move:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                history[move & 63, (move >> 6) & 63] += depth * depth
            break

    if legal == 0:
        return -MATE + ply if in_check else 0

    if best <= original_alpha:
        bound = UPPER
    elif best >= beta:
        bound = LOWER
    else:
        bound = EXACT
    existing = tt_data[slot]
    keep = (
        tt_keys[slot] == key
        and tt_generation(existing) == (ctrl[CTRL_GENERATION] & 15)
        and tt_depth(existing) > depth
    )
    if not keep:
        tt_keys[slot] = key
        tt_data[slot] = pack_tt(
            depth, bound, ctrl[CTRL_GENERATION], max(best_move, 0), to_tt_score(best, ply)
        )
    return best
```

- [ ] **Step 4: Rewrite `search.py`**

```python
"""Root of the search: time budget, iterative deepening, aspiration windows, and the
python-chess safety check around the numba kernel in search_kernel.py.

The Searcher owns every array the kernel uses and lives for one game, so the table,
killers, history and the record of root positions accumulate across our moves and can
never leak into another game.
"""

import time
from pathlib import Path
from typing import Final

import chess
import numpy as np

import bitboard as bb
import nnue_bitboard as nb
from nnue_arch import SCORE_PER_CP, WEIGHTS_FILE
from nnue_net import NetworkWeights, load_network
from search_kernel import (
    CTRL_ABORT,
    CTRL_GAME_KEYS,
    CTRL_GENERATION,
    CTRL_NODE_LIMIT,
    CTRL_NODES,
    CTRL_ROOT_HINT,
    CTRL_ROOT_MOVE,
    MATE,
    MATE_THRESHOLD,
    MAX_DEPTH,
    MAX_PLY_LIMIT,
    TT_SIZE,
    search,
)
from search_kernel import from_tt_score as _from_tt_score
from search_kernel import to_tt_score as _to_tt_score

# Time control. Every term but INCREMENT_MS comes from the clock we were handed.
MOVES_REMAINING: Final = 30  # assumed horizon; self-correcting as the clock changes
INCREMENT_MS: Final = 500  # published time control: 120 s + 0.5 s per move
SAFETY_MS: Final = 50  # margin; the referee measures wall time and does not forgive
MAX_FRACTION: Final = 0.4  # never spend more than this much of what is left
MIN_BUDGET_MS: Final = 10  # always attempt something

# Aspiration windows. From ASPIRATION_MIN_DEPTH on, the root searches a narrow window
# around the previous iteration's score; a fail outside it widens that side and retries.
ASPIRATION_MIN_DEPTH: Final = 5
ASPIRATION_WINDOW: Final = 250 * SCORE_PER_CP  # about 1.3 pawns on the net's scale

MAX_GAME_KEYS: Final = 1024


def budget_ms(time_left_ms: int) -> float:
    """Milliseconds to spend on this move, derived from the clock we were handed."""
    budget = time_left_ms / MOVES_REMAINING + 0.6 * INCREMENT_MS
    budget = min(budget, MAX_FRACTION * time_left_ms)
    return max(budget - SAFETY_MS, MIN_BUDGET_MS)


def to_tt_score(score: int, ply: int) -> int:
    return int(_to_tt_score(score, ply))


def from_tt_score(score: int, ply: int) -> int:
    return int(_from_tt_score(score, ply))


def aspiration_window(score: int, depth: int) -> tuple[int, int, int]:
    """(alpha, beta, window) to open iteration `depth` with, given the last score."""
    if depth < ASPIRATION_MIN_DEPTH:
        return -2 * MATE, 2 * MATE, 2 * MATE
    return score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW, ASPIRATION_WINDOW


def widen(alpha: int, beta: int, score: int, window: int) -> tuple[int, int, int]:
    """Open the side the search failed on, doubling the window, up to the full range."""
    window *= 2
    if score <= alpha:
        alpha = max(score - window, -2 * MATE)
    else:
        beta = min(score + window, 2 * MATE)
    return alpha, beta, window


class Searcher:
    """Alpha-beta search state for one game."""

    def __init__(self, net: NetworkWeights) -> None:
        self.net = net
        self.board = bb.new_stacks()
        self.acc = nb.new_acc_stacks()
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_data = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.full((MAX_PLY_LIMIT + 2, 2), -1, dtype=np.int32)
        self.history = np.zeros((64, 64), dtype=np.int64)
        self.moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
        self.scores = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int64)
        self.game_keys = np.zeros(MAX_GAME_KEYS, dtype=np.uint64)
        self.see_gain = np.zeros(32, dtype=np.int64)
        self.ctrl = np.zeros(8, dtype=np.int64)
        self.nodes = 0
        self._last_fullmove = 0
        self._net_tuple = (
            net.ft_w, net.ft_b, net.psqt_w, net.l1_w, net.l1_b, net.l2_w, net.l2_b,
            net.out_w, net.out_b,
        )
        self._tables = (
            self.tt_keys, self.tt_data, self.killers, self.history, self.moves, self.scores,
            self.game_keys, self.see_gain,
        )

    def note_root_position(self, board: chess.Board) -> None:
        """Record a position we were asked about, for repetition detection.

        We only ever observe positions where it is our turn, every other ply, which is
        enough: a position repeating at our turn is a genuine repetition.
        """
        if board.fullmove_number < self._last_fullmove:
            # Cannot follow the previous root. Never expected within one game; this
            # exists so a surprise cannot become a bogus draw claim.
            self.ctrl[CTRL_GAME_KEYS] = 0
        self._last_fullmove = board.fullmove_number
        bb.set_from_board(board, self.board, 0)
        n = int(self.ctrl[CTRL_GAME_KEYS])
        if n < MAX_GAME_KEYS:
            self.game_keys[n] = self.board.keys[0]
            self.ctrl[CTRL_GAME_KEYS] = n + 1

    def is_draw_key(self, key: np.uint64) -> bool:
        n = int(self.ctrl[CTRL_GAME_KEYS])
        return bool(np.any(self.game_keys[:n] == key))

    def _set_root(self, board: chess.Board) -> None:
        bb.set_from_board(board, self.board, 0)
        nb.refresh_ply(
            self.net.ft_w, self.net.ft_b, self.net.psqt_w, self.board.mailbox, self.board.state,
            0, self.acc.white_acc, self.acc.black_acc, self.acc.white_psqt, self.acc.black_psqt,
        )

    def _search(
        self, depth: int, alpha: int, beta: int, hint: int, deadline: float
    ) -> tuple[int, int]:
        self.ctrl[CTRL_ROOT_MOVE] = -1
        self.ctrl[CTRL_ROOT_HINT] = hint
        score = search(
            depth, 0, alpha, beta, True, tuple(self.board), tuple(self.acc), self._net_tuple,
            self._tables, self.ctrl, deadline,
        )
        return int(score), int(self.ctrl[CTRL_ROOT_MOVE])

    def pick(self, fen: str, time_left_ms: int, node_limit: int = 0) -> chess.Move:
        """Best move for `fen` within the budget implied by `time_left_ms`.

        `node_limit` caps the search instead of the clock when positive, which makes a
        search reproducible for tests and benchmarks.
        """
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(f"no legal moves in {fen!r}")
        self.note_root_position(board)
        self._set_root(board)
        deadline = time.monotonic() + budget_ms(time_left_ms) / 1000.0
        self.ctrl[CTRL_NODES] = 0
        self.ctrl[CTRL_ABORT] = 0
        self.ctrl[CTRL_NODE_LIMIT] = node_limit
        self.ctrl[CTRL_GENERATION] += 1

        best = -1
        score = 0
        for depth in range(1, MAX_DEPTH):
            alpha, beta, window = aspiration_window(score, depth)
            move = -1
            while True:
                value, move = self._search(depth, alpha, beta, best, deadline)
                if self.ctrl[CTRL_ABORT]:
                    break  # discard this depth entirely; it has a biased best move
                if alpha < value < beta or (alpha == -2 * MATE and beta == 2 * MATE):
                    score = value
                    break
                if value >= beta and move >= 0:
                    # A fail-high names a move that beat the window: the best lead we
                    # have if the clock cuts the re-search short. A fail-low names
                    # nothing: null-window scores are not comparable.
                    best = move
                alpha, beta, window = widen(alpha, beta, value, window)
            if self.ctrl[CTRL_ABORT]:
                break
            if move >= 0:
                best = move
            if abs(score) >= MATE_THRESHOLD:
                break  # a forced mate is as good as it gets
        self.nodes = int(self.ctrl[CTRL_NODES])

        if best >= 0:
            chosen = bb.move_to_chess(best)
            if chosen in legal:
                return chosen
            print(f"kernel proposed illegal {chosen.uci()} in {fen!r}; playing the first legal move")
        return legal[0]

    def warm_up(self) -> None:
        """Compile every kernel inside the import budget, then leave no state behind."""
        self.pick(chess.STARTING_FEN, 200)
        self.pick("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 200)
        self.tt_keys[:] = 0
        self.tt_data[:] = 0
        self.killers[:] = -1
        self.history[:] = 0
        self.ctrl[:] = 0
        self._last_fullmove = 0


def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / WEIGHTS_FILE


def load_searcher(path: Path | None = None) -> Searcher:
    return Searcher(load_network(path if path is not None else default_weights_path()))
```

`agent.py`: replace the two `nnue_engine` lines and the searcher construction with

```python
from search import load_searcher

_searcher = load_searcher()
_searcher.warm_up()
```

and update the docstring's module names (`bitboard`, `search_kernel`). `get_move` is unchanged.

- [ ] **Step 5: Run and iterate on numba typing**: `uv run pytest tests/test_search.py -q -x`. numba reports typing failures at the first call; the usual causes are a uint64 mixed with an int (wrap with `np.uint64`), a variable first assigned an int and later a numpy scalar, or a tuple element used at two types.

- [ ] **Step 6: Full suite, lint, types, commit** (add `search_kernel.py`, `search.py` to mypy files; `tests/test_parity.py` keeps passing because `nnue_engine.py` is still present):

```bash
uv run pytest -q && uv run ruff check . && uv run mypy
git add search_kernel.py search.py agent.py tests/test_search.py pyproject.toml
git commit -m "feat(search): run the alpha-beta search as a numba kernel over bitboards"
```

---

### Task 5: Import budget, speed probe, and measurements

**Files:**
- Create: `tests/test_import_time.py`
- Modify: `tools/nps.py`
- Modify: `docs/superpowers/specs/2026-09-05-bitboard-engine-design.md` (Session context)

- [ ] **Step 1: Write the import-time test**

```python
"""The platform gives 60 s before the clock starts; every kernel compiles in that window."""

import subprocess
import sys
import time


def test_agent_imports_within_half_the_platform_budget() -> None:
    start = time.monotonic()
    subprocess.run([sys.executable, "-c", "import agent"], check=True, timeout=120)
    assert time.monotonic() - start < 30
```

- [ ] **Step 2: Run**: `uv run pytest tests/test_import_time.py -q`, note the number. Over 20 s means looking at which kernel dominates with `uv run python -X importtime -c "import agent"`.

- [ ] **Step 3: Adapt `tools/nps.py`**: read the file first; replace `load_engine`/`Searcher(engine)` with `load_searcher()` and read node counts from `searcher.nodes`. Keep the four profile positions and the output format.

- [ ] **Step 4: Measure and record**: `uv run python tools/nps.py`; write knps per position and the import time into the spec's Session context under "Task 5 measurements", next to the phase 5 baseline (13.2 knps overall under load, 22 knps unloaded).

- [ ] **Step 5: Commit**

```bash
git add tests/test_import_time.py tools/nps.py docs/superpowers/specs/2026-09-05-bitboard-engine-design.md
git commit -m "test: pin the import budget and record the bitboard engine's speed"
```

---

### Task 6: Gate

- [ ] **Step 1: The `make gate` steps**: `uv run ruff check . && uv run mypy && uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000`. Expect two finished games, no flag.

- [ ] **Step 2: Self-play SPRT against the baseline**

```bash
git worktree add ../aichessathon-base 74a7d63
cp weights/nnue.npz ../aichessathon-base/weights/nnue.npz
uv run python tools/elo_bench.py --opponent ../aichessathon-base --sprt --workers 3 --pgn sprt-bitboard.pgn
git worktree remove ../aichessathon-base
```

Expect accept within a few dozen games; record the result line in the spec.

- [ ] **Step 3: Node ladder for the record**: `uv run python tools/elo_bench.py --nodes 4000 --games 20 --workers 2`, then `--nodes 16000`. Record both.

- [ ] **Step 4: Commit the record**

```bash
git add docs/superpowers/specs/2026-09-05-bitboard-engine-design.md
git commit -m "docs: record the bitboard engine gate"
```

---

### Task 7: Remove the python-chess engine

**Files:**
- Delete: `nnue_engine.py`
- Modify: `tests/test_parity.py` (pin `nnue_bitboard.evaluate` against `tests/reference.forward_int`), `tests/test_nnue_bitboard.py` (drop the `Engine` comparisons, compare against `tests.reference.forward_int` instead), `tools/verify_export.py` and `tools/export_net.py` if they import `nnue_engine`, `pyproject.toml`, `tools/TRAINING.md` file list, `CLAUDE.md` if it names the module.

- [ ] **Step 1: Find every reference**: `git grep -n nnue_engine`.
- [ ] **Step 2: Rewrite each test** so the expected value is `forward_int(_NET, board)` after `set_from_board` + `refresh_ply`, and keep the incremental-vs-refresh assertion via `update_ply`.
- [ ] **Step 3: Delete, run everything, commit**

```bash
git rm nnue_engine.py
uv run pytest -q && uv run ruff check . && uv run mypy
git add -A
git commit -m "refactor: drop the python-chess engine now that the bitboard kernels are gated"
```

- [ ] **Step 4: Build and check the zip**: `uv run python -m harness.package`; confirm it lists `agent.py bitboard.py nnue_arch.py nnue_bitboard.py nnue_features.py nnue_net.py search.py search_kernel.py weights/nnue.npz` and nothing from `tools/`. From an extracted copy run `python -c "import agent; print(agent.get_move('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 120000))"`.
