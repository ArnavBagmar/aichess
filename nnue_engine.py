"""Incremental NNUE evaluation with numba kernels.

The engine keeps one accumulator per perspective per ply in preallocated stacks and
mutates them in place; that is the entire point of NNUE (a move touches a handful of
int16 rows instead of recomputing 24,576-input matmuls), so this module trades the
project's usual immutability style for the hot path, deliberately and locally.

Scores are integers in 1/32 centipawn units, positive for the side to move, and must
match tests/reference.py forward_int exactly; tests/test_parity.py enforces that.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import numpy.typing as npt
from numba import njit

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
    NUM_PSQT_BUCKETS,
    OUTPUT_DIV,
    OUTPUT_MUL,
    WEIGHTS_FILE,
)
from nnue_features import active_features, move_deltas
from nnue_net import NetworkWeights, load_network

MAX_PLY = 256

def _jit[F: Callable[..., Any]](function: F) -> F:
    """Typed facade over numba.njit so call sites keep their signatures for mypy."""
    return cast("F", njit(cache=False)(function))


@_jit
def _refresh(
    ft_w: npt.NDArray[np.int16],
    ft_b: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    rows: npt.NDArray[np.int64],
    acc: npt.NDArray[np.int16],
    psqt: npt.NDArray[np.int32],
) -> None:
    for j in range(acc.shape[0]):
        acc[j] = ft_b[j]
    for j in range(psqt.shape[0]):
        psqt[j] = 0
    for i in range(rows.shape[0]):
        row = rows[i]
        for j in range(acc.shape[0]):
            acc[j] += ft_w[row, j]
        for j in range(psqt.shape[0]):
            psqt[j] += psqt_w[row, j]


@_jit
def _update(
    ft_w: npt.NDArray[np.int16],
    psqt_w: npt.NDArray[np.int32],
    src_acc: npt.NDArray[np.int16],
    src_psqt: npt.NDArray[np.int32],
    dst_acc: npt.NDArray[np.int16],
    dst_psqt: npt.NDArray[np.int32],
    added: npt.NDArray[np.int64],
    removed: npt.NDArray[np.int64],
) -> None:
    for j in range(dst_acc.shape[0]):
        dst_acc[j] = src_acc[j]
    for j in range(dst_psqt.shape[0]):
        dst_psqt[j] = src_psqt[j]
    for i in range(added.shape[0]):
        row = added[i]
        for j in range(dst_acc.shape[0]):
            dst_acc[j] += ft_w[row, j]
        for j in range(dst_psqt.shape[0]):
            dst_psqt[j] += psqt_w[row, j]
    for i in range(removed.shape[0]):
        row = removed[i]
        for j in range(dst_acc.shape[0]):
            dst_acc[j] -= ft_w[row, j]
        for j in range(dst_psqt.shape[0]):
            dst_psqt[j] -= psqt_w[row, j]


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


class Engine:
    """Board tracking plus incremental accumulators; one instance per game."""

    def __init__(self, net: NetworkWeights) -> None:
        self.net = net
        self.board = chess.Board()
        self._ply = 0
        self._white_acc = np.zeros((MAX_PLY, L1), dtype=np.int16)
        self._black_acc = np.zeros((MAX_PLY, L1), dtype=np.int16)
        self._white_psqt = np.zeros((MAX_PLY, NUM_PSQT_BUCKETS), dtype=np.int32)
        self._black_psqt = np.zeros((MAX_PLY, NUM_PSQT_BUCKETS), dtype=np.int32)
        self._act = np.zeros(L1, dtype=np.int64)
        self._l1c = np.zeros(L2, dtype=np.int64)
        self._l1x = np.zeros(2 * L2, dtype=np.int64)
        self._l2c = np.zeros(L3, dtype=np.int64)
        self._l2x = np.zeros(2 * L3, dtype=np.int64)
        self._refresh_ply(0)

    def set_position(self, fen: str) -> None:
        self.board = chess.Board(fen)
        self._ply = 0
        self._refresh_ply(0)

    def _refresh_ply(self, ply: int) -> None:
        for white_pov in (True, False):
            rows = np.asarray(active_features(self.board, white_pov), dtype=np.int64)
            acc = self._white_acc if white_pov else self._black_acc
            psqt = self._white_psqt if white_pov else self._black_psqt
            _refresh(self.net.ft_w, self.net.ft_b, self.net.psqt_w, rows, acc[ply], psqt[ply])

    def push(self, move: chess.Move) -> None:
        if self._ply + 1 >= MAX_PLY:
            raise OverflowError(f"accumulator stack exhausted at ply {self._ply}")
        deltas = {pov: move_deltas(self.board, move, pov) for pov in (True, False)}
        self.board.push(move)
        ply = self._ply + 1
        for white_pov in (True, False):
            acc = self._white_acc if white_pov else self._black_acc
            psqt = self._white_psqt if white_pov else self._black_psqt
            delta = deltas[white_pov]
            if delta is None:
                rows = np.asarray(active_features(self.board, white_pov), dtype=np.int64)
                _refresh(self.net.ft_w, self.net.ft_b, self.net.psqt_w, rows, acc[ply], psqt[ply])
            else:
                added, removed = delta
                _update(
                    self.net.ft_w,
                    self.net.psqt_w,
                    acc[ply - 1],
                    psqt[ply - 1],
                    acc[ply],
                    psqt[ply],
                    np.asarray(added, dtype=np.int64),
                    np.asarray(removed, dtype=np.int64),
                )
        self._ply = ply

    def pop(self) -> None:
        if self._ply == 0:
            raise IndexError("pop with no pushed moves")
        self.board.pop()
        self._ply -= 1

    def evaluate(self) -> int:
        """Score in 1/32 centipawn units, positive for the side to move."""
        ply = self._ply
        bucket = (chess.popcount(self.board.occupied) - 1) // 4
        white_to_move = self.board.turn == chess.WHITE
        psqt_diff = int(self._white_psqt[ply, bucket]) - int(self._black_psqt[ply, bucket])
        stm_acc = self._white_acc[ply] if white_to_move else self._black_acc[ply]
        ntm_acc = self._black_acc[ply] if white_to_move else self._white_acc[ply]
        return _forward(
            stm_acc,
            ntm_acc,
            psqt_diff,
            white_to_move,
            self.net.l1_w[bucket],
            self.net.l1_b[bucket],
            self.net.l2_w[bucket],
            self.net.l2_b[bucket],
            self.net.out_w[bucket],
            int(self.net.out_b[bucket]),
            self._act,
            self._l1c,
            self._l1x,
            self._l2c,
            self._l2x,
        )

    def evaluate_cp(self) -> float:
        return self.evaluate() / 32.0


def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / WEIGHTS_FILE


def load_engine(path: Path | None = None) -> Engine:
    return Engine(load_network(path if path is not None else default_weights_path()))


def warm_up(engine: Engine) -> None:
    """Trigger every numba compilation now, inside the platform's init budget.

    Covers all three kernels with real argument types: a refresh (set_position), an
    incremental update (quiet pawn push), a king-move refresh, and a forward pass.
    """
    engine.set_position(chess.STARTING_FEN)
    engine.push(chess.Move.from_uci("e2e4"))
    engine.push(chess.Move.from_uci("e7e5"))
    engine.push(chess.Move.from_uci("e1e2"))
    engine.evaluate()
    for _ in range(3):
        engine.pop()
