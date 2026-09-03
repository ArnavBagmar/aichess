"""Quantized network weights: container, npz loading, and shape validation.

The .npz layout is our own (tools/gen_random_net.py writes it today; the Phase 2
exporter will write the same keys from a trained nnue-pytorch checkpoint):

  ft_w   int16 [NUM_FEATURES, L1]   feature transformer, 1/256 units
  ft_b   int16 [L1]                 feature transformer bias, 1/256 units
  psqt_w int32 [NUM_FEATURES, 8]    PSQT columns, 1/9600 units
  l1_w   int8  [8, L2, L1]          per-bucket first layer, 1/128 units
  l1_b   int32 [8, L2]              1/16384 units
  l2_w   int8  [8, L3, 2*L2]        1/64 units
  l2_b   int32 [8, L3]              1/8192 units
  out_w  int8  [8, 2*L2 + 2*L3]     1/128 units
  out_b  int32 [8]                  1/16384 units
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nnue_arch import (
    L1,
    L2,
    L3,
    MAX_ACTIVE_FEATURES,
    NUM_FEATURES,
    NUM_LS_BUCKETS,
    NUM_PSQT_BUCKETS,
    OUT_IN,
)

_EXPECTED: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
    "ft_w": ((NUM_FEATURES, L1), np.int16),
    "ft_b": ((L1,), np.int16),
    "psqt_w": ((NUM_FEATURES, NUM_PSQT_BUCKETS), np.int32),
    "l1_w": ((NUM_LS_BUCKETS, L2, L1), np.int8),
    "l1_b": ((NUM_LS_BUCKETS, L2), np.int32),
    "l2_w": ((NUM_LS_BUCKETS, L3, 2 * L2), np.int8),
    "l2_b": ((NUM_LS_BUCKETS, L3), np.int32),
    "out_w": ((NUM_LS_BUCKETS, OUT_IN), np.int8),
    "out_b": ((NUM_LS_BUCKETS,), np.int32),
}


@dataclass(frozen=True)
class NetworkWeights:
    """The quantized parameters, exactly as stored on disk."""

    ft_w: npt.NDArray[np.int16]
    ft_b: npt.NDArray[np.int16]
    psqt_w: npt.NDArray[np.int32]
    l1_w: npt.NDArray[np.int8]
    l1_b: npt.NDArray[np.int32]
    l2_w: npt.NDArray[np.int8]
    l2_b: npt.NDArray[np.int32]
    out_w: npt.NDArray[np.int8]
    out_b: npt.NDArray[np.int32]


def load_network(path: Path) -> NetworkWeights:
    """Load and validate a weights archive; raise ValueError on any mismatch."""
    if not path.is_file():
        raise ValueError(f"network weights not found at {path}")
    with np.load(path) as archive:
        missing = sorted(set(_EXPECTED) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        arrays: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in _EXPECTED.items():
            array = archive[name]
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"{path}: {name} has shape {array.shape} dtype {array.dtype}, "
                    f"expected {shape} {np.dtype(dtype)}"
                )
            arrays[name] = np.ascontiguousarray(array)
    _check_accumulator_headroom(path, arrays["ft_w"], arrays["ft_b"])
    return NetworkWeights(**arrays)


def _check_accumulator_headroom(
    path: Path, ft_w: npt.NDArray[np.int16], ft_b: npt.NDArray[np.int16]
) -> None:
    """Reject weights that could silently wrap the int16 accumulator.

    The engine sums at most MAX_ACTIVE_FEATURES rows plus the bias per column; the
    kernels have no overflow check (int16 wraps two's-complement with no signal), so
    the bound is enforced once here, at load time. Conservative by design: it assumes
    the worst column value repeats for every active feature.
    """
    worst = (
        np.abs(ft_w.astype(np.int64)).max(axis=0) * MAX_ACTIVE_FEATURES
        + np.abs(ft_b.astype(np.int64))
    )
    limit = int(np.iinfo(np.int16).max)
    if int(worst.max()) > limit:
        column = int(worst.argmax())
        raise ValueError(
            f"{path}: feature transformer column {column} can reach {int(worst.max())}, "
            f"beyond the int16 accumulator limit of {limit}"
        )
