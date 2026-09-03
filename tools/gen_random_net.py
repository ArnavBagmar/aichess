"""Generate a randomly initialized quantized network for pipeline testing.

Phase 1 has no trained checkpoint yet; this writes weights/nnue.npz with seeded random
integers whose magnitudes keep every activation stage away from full saturation, so the
parity tests exercise interior code paths rather than clipped constants. The Phase 2
exporter replaces this file's output with real trained weights in the same layout.

Usage: uv run python tools/gen_random_net.py [--seed N] [--out weights/nnue.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnue_arch import (
    L1,
    L2,
    L3,
    NUM_FEATURES,
    NUM_LS_BUCKETS,
    NUM_PSQT_BUCKETS,
    OUT_IN,
)
from nnue_net import NetworkWeights, load_network


def random_network(seed: int) -> NetworkWeights:
    rng = np.random.default_rng(seed)
    return NetworkWeights(
        ft_w=rng.integers(-48, 49, (NUM_FEATURES, L1)).astype(np.int16),
        ft_b=rng.integers(0, 301, (L1,)).astype(np.int16),
        psqt_w=rng.integers(-40_000, 40_001, (NUM_FEATURES, NUM_PSQT_BUCKETS)).astype(np.int32),
        l1_w=rng.integers(-64, 65, (NUM_LS_BUCKETS, L2, L1)).astype(np.int8),
        l1_b=rng.integers(-16_384, 16_385, (NUM_LS_BUCKETS, L2)).astype(np.int32),
        l2_w=rng.integers(-64, 65, (NUM_LS_BUCKETS, L3, 2 * L2)).astype(np.int8),
        l2_b=rng.integers(-8_192, 8_193, (NUM_LS_BUCKETS, L3)).astype(np.int32),
        out_w=rng.integers(-127, 128, (NUM_LS_BUCKETS, OUT_IN)).astype(np.int8),
        out_b=rng.integers(-16_384, 16_385, (NUM_LS_BUCKETS,)).astype(np.int32),
    )


def save_network(net: NetworkWeights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ft_w=net.ft_w,
        ft_b=net.ft_b,
        psqt_w=net.psqt_w,
        l1_w=net.l1_w,
        l1_b=net.l1_b,
        l2_w=net.l2_w,
        l2_b=net.l2_b,
        out_w=net.out_w,
        out_b=net.out_b,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("weights") / "nnue.npz")
    arguments = parser.parse_args()

    save_network(random_network(arguments.seed), arguments.out)
    load_network(arguments.out)  # round-trip validation
    size = arguments.out.stat().st_size
    print(f"wrote {arguments.out} ({size:,} bytes, seed {arguments.seed})")


if __name__ == "__main__":
    main()
