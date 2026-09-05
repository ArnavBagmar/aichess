"""Quantize the trained residual HalfKP evaluator for fast submission inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--activation-scale", type=int, default=256)
    parser.add_argument("--output-scale", type=int, default=4096)
    args = parser.parse_args()

    source = np.load(args.source)
    embedding = np.rint(
        source["embedding"].astype(np.float32) * args.activation_scale
    ).astype(np.int16)
    hidden_bias = np.rint(
        source["hidden_bias"].astype(np.float32) * args.activation_scale
    ).astype(np.int16)
    output = np.rint(source["output"].astype(np.float32) * args.output_scale).astype(
        np.int16
    )
    output_bias_cp = np.rint(
        source["output_bias"].astype(np.float32) * 400.0
    ).astype(np.int16)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.destination,
        embedding=embedding,
        hidden_bias=hidden_bias,
        output=output,
        output_bias_cp=output_bias_cp,
        activation_scale=np.asarray([args.activation_scale], dtype=np.int32),
        output_scale=np.asarray([args.output_scale], dtype=np.int32),
    )
    print(f"wrote {args.destination} ({args.destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
