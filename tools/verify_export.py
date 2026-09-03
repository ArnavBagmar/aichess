"""End-to-end export verification: trainer forward vs our engine.

Runs in the nnue-pytorch trainer virtualenv (needs the `model` package and torch).
It loads a trained checkpoint, exports it through export_net, and checks that our
integer engine reproduces the trainer's own forward pass. The trainer is put in the
exact configuration that makes fake-quantized training math equal true integer
inference: float64 weights and FAKE_QUANTIZE_EPS patched to 0.0.

A mismatch here means export_net misread the checkpoint (wrong key, un-merged
factorization, wrong scale) or the architecture constants drifted — the two failure
modes that produce a silently weaker agent.

Usage (from the starter repo root, with the trainer venv's python):
    TRAINER=~/Desktop/aichessathon-train/nnue-pytorch \
    PYTHONPATH=$TRAINER \
    $TRAINER/.venv/bin/python tools/verify_export.py <checkpoint.ckpt> \
        --trainer $TRAINER [--positions 300]
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nnue_arch import L1, NNUE2SCORE, SCORE_PER_CP  # noqa: E402
from nnue_engine import Engine  # noqa: E402
from nnue_features import active_features  # noqa: E402
from nnue_net import load_network  # noqa: E402
from tools.export_net import export  # noqa: E402

MAX_ACTIVE = 32


def _index_tensor(board: chess.Board, white_pov: bool) -> torch.Tensor:
    features = active_features(board, white_pov)
    padded = features + [-1] * (MAX_ACTIVE - len(features))
    return torch.tensor(padded, dtype=torch.int32).unsqueeze(0)


def trainer_forward(model: torch.nn.Module, board: chess.Board) -> float:
    """The trainer's own fake-quantized forward for one board, in 1/32 cp units."""
    us = torch.tensor([[1.0 if board.turn == chess.WHITE else 0.0]], dtype=torch.float64)
    them = 1.0 - us
    white = _index_tensor(board, True)
    black = _index_tensor(board, False)
    piece_count = torch.tensor([chess.popcount(board.occupied)], dtype=torch.int64)
    with torch.no_grad():
        value = model(us, them, white, black, piece_count)
    return float(value.item()) * NNUE2SCORE * SCORE_PER_CP


def build_model(trainer_dir: Path, checkpoint: Path) -> torch.nn.Module:
    import model as trainer_model  # noqa: PLC0415
    from model import quantize  # noqa: PLC0415

    quantize.FAKE_QUANTIZE_EPS = 0.0  # exact integer emulation, not the training epsilon

    config = trainer_model.NNUELightningConfig(features="HalfKAv2_hm^")
    config.model_config.L1 = L1
    nnue = trainer_model.NNUE(config=config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    nnue.load_state_dict(payload["state_dict"])
    inner = nnue.model
    inner.double()
    inner.eval()
    return inner


def sample_boards(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = [chess.Board()]
    while len(boards) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
            boards.append(board.copy(stack=False))
    return boards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=300)
    arguments = parser.parse_args()

    model = build_model(arguments.trainer, arguments.checkpoint)

    export_path = REPO / "weights" / "_verify.npz"
    net = export(arguments.checkpoint)
    np.savez_compressed(export_path, **{f: getattr(net, f) for f in net.__dataclass_fields__})
    engine = Engine(load_network(export_path))

    boards = sample_boards(arguments.positions, seed=1)
    diffs = []
    for board in boards:
        engine.set_position(board.fen())
        expected = trainer_forward(model, board)
        diffs.append(abs(engine.evaluate() - expected))

    diffs_arr = np.array(diffs)
    within_1cp = int((diffs_arr <= SCORE_PER_CP).sum())
    print(f"positions:      {len(boards)}")
    print(f"max diff:       {diffs_arr.max():.1f} internal units "
          f"({diffs_arr.max() / SCORE_PER_CP:.2f} cp)")
    print(f"mean diff:      {diffs_arr.mean():.2f} internal units")
    print(f"within 1 cp:    {within_1cp}/{len(boards)}")
    export_path.unlink(missing_ok=True)

    # eps=0 + float64 should make the trainer forward exactly integer; allow a hair
    # for the trunc-vs-round boundary in the final scaling.
    if diffs_arr.max() > 2 * SCORE_PER_CP:
        raise SystemExit("FAIL: export does not reproduce the trainer forward pass")
    print("PASS")


if __name__ == "__main__":
    main()
