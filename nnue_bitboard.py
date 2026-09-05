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
    """Accumulators per ply per perspective, plus the forward pass's scratch space."""

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
    return int((sq ^ horizontal ^ vertical) + NUM_SQ * plane + NUM_PLANES * bucket)


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
    """Rebuild both perspectives' accumulators at `ply` from the mailbox."""
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
    """Accumulators for ply + 1 after `move`, which make_move has already applied."""
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
    half = L1 // 2

    for i in range(half):
        a = min(max(int(stm_acc[i]), 0), FT_ACT_MAX)
        b = min(max(int(stm_acc[half + i]), 0), FT_ACT_MAX)
        act[i] = (a * b) >> FT_PAIRWISE_SHIFT
        c = min(max(int(ntm_acc[i]), 0), FT_ACT_MAX)
        d = min(max(int(ntm_acc[half + i]), 0), FT_ACT_MAX)
        act[half + i] = (c * d) >> FT_PAIRWISE_SHIFT

    for j in range(L2):
        total = np.int64(l1_b[j])
        for i in range(L1):
            total += np.int64(l1_w[j, i]) * act[i]
        l1c[j] = total
    skip = l1c[L2 - 2] - l1c[L2 - 1]
    for j in range(L2):
        l1x[j] = min((l1c[j] * l1c[j]) >> L1_SQUARE_SHIFT, HIDDEN_ACT_MAX)
        l1x[L2 + j] = min(max(l1c[j] >> L1_LINEAR_SHIFT, 0), HIDDEN_ACT_MAX)

    for j in range(L3):
        total = np.int64(l2_b[j])
        for i in range(2 * L2):
            total += np.int64(l2_w[j, i]) * l1x[i]
        l2c[j] = total
    for j in range(L3):
        l2x[j] = min((l2c[j] * l2c[j]) >> L2_SQUARE_SHIFT, HIDDEN_ACT_MAX)
        l2x[L3 + j] = min(max(l2c[j] >> L2_LINEAR_SHIFT, 0), HIDDEN_ACT_MAX)

    out = np.int64(out_b)
    for i in range(2 * L2):
        out += np.int64(out_w[i]) * l1x[i]
    for i in range(2 * L3):
        out += np.int64(out_w[2 * L2 + i]) * l2x[i]

    # Trunc-toward-zero division, matching the trainer's rounding_mode="trunc".
    numerator = (out + skip) * OUTPUT_MUL
    quantized = numerator // OUTPUT_DIV if numerator >= 0 else -(-numerator // OUTPUT_DIV)

    if white_to_move:
        return int(2 * quantized + psqt_diff)
    return int(2 * quantized - psqt_diff)


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
            white_acc[ply],
            black_acc[ply],
            psqt_diff,
            True,
            l1_w[bucket],
            l1_b[bucket],
            l2_w[bucket],
            l2_b[bucket],
            out_w[bucket],
            int(out_b[bucket]),
            act,
            l1c,
            l1x,
            l2c,
            l2x,
        )
    return _forward(
        black_acc[ply],
        white_acc[ply],
        psqt_diff,
        False,
        l1_w[bucket],
        l1_b[bucket],
        l2_w[bucket],
        l2_b[bucket],
        out_w[bucket],
        int(out_b[bucket]),
        act,
        l1c,
        l1x,
        l2c,
        l2x,
    )
