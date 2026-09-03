"""Validate export_net's checkpoint mapping without the trainer package.

Builds a synthetic state_dict with exactly the keys, shapes, and factorization
structure nnue-pytorch writes for HalfKAv2_hm^ / L1=256, saves it as a checkpoint,
exports it, then dequantizes the exported integers and checks they match the
original merged float weights within one quantization step. This catches the export
failure modes that matter: a wrong key, an un-merged factorization/virtual weight, a
wrong scale, or a bad reshape. It does not exercise the trainer's forward code
(tools/verify_export.py does that in the trainer venv).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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

FT_COLS = L1 + NUM_PSQT_BUCKETS


def _synthetic_state_dict(seed: int) -> dict[str, "torch.Tensor"]:
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float) -> "torch.Tensor":
        return (torch.rand(*shape, generator=gen, dtype=torch.float64) - 0.5) * 2 * scale

    return {
        # Feature transformer: small magnitudes so nothing saturates on quantize.
        "model.input.features.0.weight": rand(NUM_FEATURES, FT_COLS, scale=0.4),
        "model.input.features.0.virtual_weight": rand(NUM_PLANES, FT_COLS, scale=0.1),
        "model.input.bias": rand(FT_COLS, scale=0.4),
        # Layer stacks. l1 is factorized; l2 and output are not.
        "model.layer_stacks.l1.linear.weight": rand(NUM_LS_BUCKETS * L2, L1, scale=0.3),
        "model.layer_stacks.l1.linear.bias": rand(NUM_LS_BUCKETS * L2, scale=0.3),
        "model.layer_stacks.l1.factorized_linear.weight": rand(L2, L1, scale=0.05),
        "model.layer_stacks.l1.factorized_linear.bias": rand(L2, scale=0.05),
        "model.layer_stacks.l2.linear.weight": rand(NUM_LS_BUCKETS * L3, 2 * L2, scale=0.5),
        "model.layer_stacks.l2.linear.bias": rand(NUM_LS_BUCKETS * L3, scale=0.5),
        "model.layer_stacks.output.linear.weight": rand(NUM_LS_BUCKETS * 1, OUT_IN, scale=0.3),
        "model.layer_stacks.output.linear.bias": rand(NUM_LS_BUCKETS * 1, scale=0.3),
    }


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, dict[str, np.ndarray]]:
    from tools.export_net import export  # noqa: PLC0415

    state = _synthetic_state_dict(3)
    ckpt = tmp_path_factory.mktemp("ckpt") / "net.ckpt"
    torch.save({"state_dict": state}, ckpt)
    return export(Path(ckpt)), {k: v.numpy() for k, v in state.items()}


def _close(actual: np.ndarray, expected: np.ndarray, scale: float, label: str) -> None:
    # Dequantized value should be within half a quantization step of the original.
    assert np.abs(actual - expected).max() <= 1.0 / scale + 1e-9, label


def test_feature_transformer_merges_virtual(exported: tuple) -> None:
    net, state = exported
    merged = state["model.input.features.0.weight"] + np.tile(
        state["model.input.features.0.virtual_weight"], (NUM_KING_BUCKETS, 1)
    )
    _close(net.ft_w.astype(np.float64) / 256.0, merged[:, :L1], 256.0, "ft_w")
    _close(net.psqt_w.astype(np.float64) / 9600.0, merged[:, L1:], 9600.0, "psqt_w")
    _close(net.ft_b.astype(np.float64) / 256.0, state["model.input.bias"][:L1], 256.0, "ft_b")


def test_l1_merges_factorization(exported: tuple) -> None:
    net, state = exported
    merged = state["model.layer_stacks.l1.linear.weight"] + np.tile(
        state["model.layer_stacks.l1.factorized_linear.weight"], (NUM_LS_BUCKETS, 1)
    )
    expected = merged.reshape(NUM_LS_BUCKETS, L2, L1)
    _close(net.l1_w.astype(np.float64) / 128.0, expected, 128.0, "l1_w")


def test_l2_and_output_shapes_and_scales(exported: tuple) -> None:
    net, state = exported
    l2 = state["model.layer_stacks.l2.linear.weight"].reshape(NUM_LS_BUCKETS, L3, 2 * L2)
    _close(net.l2_w.astype(np.float64) / 64.0, l2, 64.0, "l2_w")
    out = state["model.layer_stacks.output.linear.weight"].reshape(NUM_LS_BUCKETS, OUT_IN)
    _close(net.out_w.astype(np.float64) / 128.0, out, 128.0, "out_w")
    assert net.out_b.shape == (NUM_LS_BUCKETS,)


def test_exported_net_loads_and_evaluates(exported: tuple) -> None:
    import chess  # noqa: PLC0415

    from nnue_engine import Engine  # noqa: PLC0415

    net, _ = exported
    engine = Engine(net)
    engine.set_position(chess.STARTING_FEN)
    score = engine.evaluate()
    assert isinstance(score, int)
