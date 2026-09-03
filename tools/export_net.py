"""Export a trained nnue-pytorch checkpoint to weights/nnue.npz.

Reads the raw state_dict (no nnue-pytorch import needed), coalesces the training-time
factorizations exactly like the trainer does at serialization, applies the trainer's
quantization scales, and writes the archive nnue_net.load_network expects:

  feature transformer   merged = weight + virtual_weight tiled over king buckets;
                        columns [0, L1) scale 256 -> int16, PSQT columns scale 9600 -> int32
  layer stacks          l1 merged = linear + factorized_linear tiled over buckets;
                        weights int8 (scales 128 / 64 / 128), biases int32
                        (scales 16384 / 8192 / 16384)

Every tensor is located by key suffix so trainer-side renames fail loudly here instead
of silently exporting garbage. Overflow anywhere aborts the export.

Usage: python tools/export_net.py path/to/last.ckpt [--out weights/nnue.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnue_arch import (  # noqa: E402
    L1,
    L2,
    L3,
    NUM_FEATURES,
    NUM_KING_BUCKETS,
    NUM_LS_BUCKETS,
    NUM_PLANES,
    NUM_PSQT_BUCKETS,
    OUT_IN,
)
from nnue_net import NetworkWeights, load_network  # noqa: E402
from tools.gen_random_net import save_network  # noqa: E402

FT_COLS = L1 + NUM_PSQT_BUCKETS


def _find(state: dict[str, torch.Tensor], suffix: str, shape: tuple[int, ...]) -> torch.Tensor:
    matches = [key for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one state_dict key ending in {suffix!r}, "
                         f"found {matches!r}")
    tensor = state[matches[0]].detach().to(torch.float64)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{matches[0]} has shape {tuple(tensor.shape)}, expected {shape}")
    return tensor


def _quantize(
    values: torch.Tensor, scale: int, dtype: type[np.generic], label: str
) -> np.ndarray:
    scaled = torch.round(values * scale)
    info = np.iinfo(dtype)
    low, high = float(-info.max), float(info.max)
    outside = int(((scaled < low) | (scaled > high)).sum())
    if outside:
        raise ValueError(
            f"{label}: {outside} values outside [{low:.0f}, {high:.0f}] after scaling by "
            f"{scale}; the checkpoint does not fit the quantization scheme"
        )
    return scaled.numpy().astype(dtype)


def export(checkpoint_path: Path) -> NetworkWeights:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state: dict[str, torch.Tensor] = (
        payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    )
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path} does not contain a state_dict")

    ft = _find(state, "input.features.0.weight", (NUM_FEATURES, FT_COLS))
    virtual = _find(state, "input.features.0.virtual_weight", (NUM_PLANES, FT_COLS))
    merged_ft = ft + virtual.repeat(NUM_KING_BUCKETS, 1)
    ft_bias = _find(state, "input.bias", (FT_COLS,))

    def stacked(
        prefix: str, out_features: int, in_features: int, factorized: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = _find(state, f"{prefix}.linear.weight",
                       (NUM_LS_BUCKETS * out_features, in_features))
        bias = _find(state, f"{prefix}.linear.bias", (NUM_LS_BUCKETS * out_features,))
        if factorized:
            weight = weight + _find(
                state, f"{prefix}.factorized_linear.weight", (out_features, in_features)
            ).repeat(NUM_LS_BUCKETS, 1)
            bias = bias + _find(
                state, f"{prefix}.factorized_linear.bias", (out_features,)
            ).repeat(NUM_LS_BUCKETS)
        return weight.reshape(NUM_LS_BUCKETS, out_features, in_features), bias.reshape(
            NUM_LS_BUCKETS, out_features
        )

    l1_w, l1_b = stacked("layer_stacks.l1", L2, L1, factorized=True)
    l2_w, l2_b = stacked("layer_stacks.l2", L3, 2 * L2, factorized=False)
    out_w, out_b = stacked("layer_stacks.output", 1, OUT_IN, factorized=False)

    return NetworkWeights(
        ft_w=_quantize(merged_ft[:, :L1], 256, np.int16, "ft_w"),
        ft_b=_quantize(ft_bias[:L1], 256, np.int16, "ft_b"),
        psqt_w=_quantize(merged_ft[:, L1:], 9600, np.int32, "psqt_w"),
        l1_w=_quantize(l1_w, 128, np.int8, "l1_w"),
        l1_b=_quantize(l1_b, 128 * 128, np.int32, "l1_b"),
        l2_w=_quantize(l2_w, 64, np.int8, "l2_w"),
        l2_b=_quantize(l2_b, 64 * 128, np.int32, "l2_b"),
        out_w=_quantize(out_w.reshape(NUM_LS_BUCKETS, OUT_IN), 128, np.int8, "out_w"),
        out_b=_quantize(out_b.reshape(NUM_LS_BUCKETS), 128 * 128, np.int32, "out_b"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("weights") / "nnue.npz")
    arguments = parser.parse_args()

    net = export(arguments.checkpoint)
    save_network(net, arguments.out)
    load_network(arguments.out)  # re-validate, including accumulator headroom
    print(f"exported {arguments.checkpoint} -> {arguments.out} "
          f"({arguments.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
