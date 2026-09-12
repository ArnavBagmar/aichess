"""Loader validation: round-trip, shape/dtype rejection, and overflow headroom."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nnue_net import load_network
from tools.gen_random_net import random_network, save_network


def test_round_trip(tmp_path: Path) -> None:
    net = random_network(5)
    path = tmp_path / "net.npz"
    save_network(net, path)
    loaded = load_network(path)
    assert np.array_equal(loaded.ft_w, net.ft_w)
    assert np.array_equal(loaded.out_b, net.out_b)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_network(tmp_path / "absent.npz")


def test_wrong_shape_rejected(tmp_path: Path) -> None:
    net = random_network(5)
    bad = replace(net, ft_b=net.ft_b[:-1])
    path = tmp_path / "net.npz"
    save_network(bad, path)
    with pytest.raises(ValueError, match="ft_b"):
        load_network(path)


def test_accumulator_overflow_rejected(tmp_path: Path) -> None:
    net = random_network(5)
    hot = net.ft_w.copy()
    hot[:, 7] = 2_000  # 32 * 2000 blows well past int16
    path = tmp_path / "net.npz"
    save_network(replace(net, ft_w=hot), path)
    with pytest.raises(ValueError, match="column 7"):
        load_network(path)
