"""Slow, obviously-correct reference implementations of the network forward pass.

Two references with different jobs:

- forward_int: the integer pipeline the engine must reproduce bit-for-bit. Pure numpy,
  int64 everywhere except the deliberate int16 cast of the accumulator (the engine
  stores accumulators as int16, exactly like Stockfish).
- forward_float: a float64 transcription of nnue-pytorch's fake-quantized training
  forward (layer_stacks.py + composed_feature_transformer.py + quantize.py). The trainer
  adds 1e-5 before floor on activations, so this can differ from forward_int by a unit or
  two on rare boundary values; tests bound that difference instead of demanding equality.

Both return the score in 1/32 centipawn units, positive for the side to move.
"""

import chess
import numpy as np
import numpy.typing as npt

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
    OUTPUT_DIV,
    OUTPUT_MUL,
)
from nnue_features import active_features
from nnue_net import NetworkWeights

_EPS = 1e-5  # FAKE_QUANTIZE_EPS in the trainer


def bucket_of(board: chess.Board) -> int:
    return (chess.popcount(board.occupied) - 1) // 4


def accumulate_int(
    net: NetworkWeights, features: list[int]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Accumulator (int16 semantics, returned widened) and PSQT accumulator."""
    rows = np.asarray(features, dtype=np.int64)
    acc = net.ft_w[rows].astype(np.int64).sum(axis=0) + net.ft_b.astype(np.int64)
    if acc.max() > np.iinfo(np.int16).max or acc.min() < np.iinfo(np.int16).min:
        raise OverflowError("feature transformer accumulator overflows int16")
    psqt = net.psqt_w[rows].astype(np.int64).sum(axis=0)
    return acc.astype(np.int16).astype(np.int64), psqt


def _ft_activation(acc: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    clipped = np.clip(acc, 0, FT_ACT_MAX)
    half = L1 // 2
    result: npt.NDArray[np.int64] = (clipped[:half] * clipped[half:]) >> FT_PAIRWISE_SHIFT
    return result


def _trunc_div(numerator: int, denominator: int) -> int:
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def forward_int(net: NetworkWeights, board: chess.Board) -> int:
    """Integer forward pass; ground truth for the numba engine."""
    white_acc, white_psqt = accumulate_int(net, active_features(board, True))
    black_acc, black_psqt = accumulate_int(net, active_features(board, False))
    white_to_move = board.turn == chess.WHITE
    stm, ntm = (white_acc, black_acc) if white_to_move else (black_acc, white_acc)
    bucket = bucket_of(board)

    act = np.concatenate([_ft_activation(stm), _ft_activation(ntm)])

    l1c = net.l1_b[bucket].astype(np.int64) + net.l1_w[bucket].astype(np.int64) @ act
    skip = int(l1c[L2 - 2] - l1c[L2 - 1])
    l1x = np.concatenate(
        [
            np.clip((l1c * l1c) >> L1_SQUARE_SHIFT, 0, HIDDEN_ACT_MAX),
            np.clip(l1c >> L1_LINEAR_SHIFT, 0, HIDDEN_ACT_MAX),
        ]
    )

    l2c = net.l2_b[bucket].astype(np.int64) + net.l2_w[bucket].astype(np.int64) @ l1x
    l2x = np.concatenate(
        [
            np.clip((l2c * l2c) >> L2_SQUARE_SHIFT, 0, HIDDEN_ACT_MAX),
            np.clip(l2c >> L2_LINEAR_SHIFT, 0, HIDDEN_ACT_MAX),
        ]
    )

    out_in = np.concatenate([l1x, l2x])
    out = int(net.out_b[bucket]) + int(net.out_w[bucket].astype(np.int64) @ out_in)
    quantized = _trunc_div((out + skip) * OUTPUT_MUL, OUTPUT_DIV)

    psqt_diff = int(white_psqt[bucket] - black_psqt[bucket])
    return 2 * quantized + (psqt_diff if white_to_move else -psqt_diff)


def _fake_floor(values: npt.NDArray[np.float64], scale: float) -> npt.NDArray[np.float64]:
    """The trainer's _fake_quantize_acts: floor(x * scale + eps) / scale."""
    result: npt.NDArray[np.float64] = np.floor(values * scale + _EPS) / scale
    return result


def forward_float(net: NetworkWeights, board: chess.Board) -> float:
    """Float64 transcription of the trainer's fake-quantized forward pass."""
    ft_w = net.ft_w.astype(np.float64) / 256.0
    ft_b = net.ft_b.astype(np.float64) / 256.0
    psqt_w = net.psqt_w.astype(np.float64) / 9600.0

    def perspective(white_pov: bool) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rows = np.asarray(active_features(board, white_pov), dtype=np.int64)
        return ft_w[rows].sum(axis=0) + ft_b, psqt_w[rows].sum(axis=0)

    white, wpsqt_all = perspective(True)
    black, bpsqt_all = perspective(False)
    white_to_move = board.turn == chess.WHITE
    bucket = bucket_of(board)
    wpsqt, bpsqt = wpsqt_all[bucket], bpsqt_all[bucket]

    stacked = np.concatenate([white, black] if white_to_move else [black, white])
    stacked = np.clip(stacked, 0.0, 255.0 / 256.0)
    quarters = np.split(stacked, 4)
    l0 = np.concatenate([quarters[0] * quarters[1], quarters[2] * quarters[3]])
    l0 = _fake_floor(l0, 128.0)  # l0_correction_factor is exactly 1.0

    def stack_layer(
        weight: npt.NDArray[np.int8],
        bias: npt.NDArray[np.int32],
        weight_scale: float,
        bias_scale: float,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        w = weight[bucket].astype(np.float64) / weight_scale
        b = bias[bucket].astype(np.float64) / bias_scale
        result: npt.NDArray[np.float64] = w @ x + b
        return result

    l1c = stack_layer(net.l1_w, net.l1_b, 128.0, 16384.0, l0)
    skip = float(l1c[L2 - 2] - l1c[L2 - 1])
    l1x = np.clip(
        np.concatenate([_fake_floor(l1c * l1c, 128.0), _fake_floor(l1c, 128.0)]),
        0.0,
        127.0 / 128.0,
    )

    l2c = stack_layer(net.l2_w, net.l2_b, 64.0, 8192.0, l1x)
    l2x = np.clip(
        np.concatenate([_fake_floor(l2c * l2c, 128.0), _fake_floor(l2c, 128.0)]),
        0.0,
        127.0 / 128.0,
    )

    out_w = net.out_w[bucket].astype(np.float64) / 128.0
    out_b = float(net.out_b[bucket]) / 16384.0
    l3 = float(out_w @ np.concatenate([l1x, l2x])) + out_b + skip

    # fake_quantize_output: round to the 1/32768 grid, then trunc-scale to 1/9600.
    fwd = float(np.rint(l3 * 32768.0))
    value = float(np.trunc(fwd * 9600.0 / 32768.0))

    x = value / 9600.0 + float(wpsqt - bpsqt) * (0.5 if white_to_move else -0.5)
    return x * 19200.0  # nnue2score * SCORE_PER_CP -> same units as forward_int
