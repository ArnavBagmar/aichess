# Training and exporting the NNUE (Phase 2)

The competition zip ships only `agent.py`, the `nnue_*.py` modules, and
`weights/nnue.npz`. Training happens outside this repo, in a clone of
`official-stockfish/nnue-pytorch`, and produces a checkpoint that
`tools/export_net.py` converts into the `.npz` the agent loads. Nothing from the
trainer is shipped — only the weights we trained, which the rules allow.

## One-time setup (trainer workspace)

```bash
git clone https://github.com/official-stockfish/nnue-pytorch ~/aichessathon-train/nnue-pytorch
cd ~/aichessathon-train/nnue-pytorch
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt torch

# Build the C++ data loader (never shipped; the no-native-binaries rule is about
# the submission zip, not the trainer).
cmake -S data_loader/cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

Training data: any `.binpack` from the nnue-pytorch training-data sets
(huggingface `linrock/*`, robotmoon.com index). A single few-GB shard is enough for
a first net; download the `.zst` and `zstd -d` it.

## Train

Restrict to the classic feature set and our layer size so the checkpoint matches
`nnue_arch.py` (feature set `HalfKAv2_hm^`, `--l1 256`). Booleans are tyro flags:
`--no-pin-memory`, not `--pin-memory False`.

```bash
.venv/bin/python train.py <data.binpack> \
  --features "HalfKAv2_hm^" --l1 256 \
  --accelerator mps --no-pin-memory --num-workers 1 \
  --batch-size 16384 --max-epochs 400 \
  --default-root-dir runs/net1 --network-save-period 20
```

Gotcha: on macOS the data loader can deadlock at 0% CPU when `pin_memory` is on or
with multiple workers under MPS. If a run stalls with no batch progress, kill it and
re-run with `--no-pin-memory --num-workers 1`; drop to `--accelerator cpu` to take
MPS out of the picture entirely for a smoke test.

## Export

From the starter repo root, with the trainer on `PYTHONPATH`:

```bash
python tools/export_net.py runs/net1/lightning_logs/version_0/checkpoints/last.ckpt
# -> weights/nnue.npz
```

`export_net.py` needs only torch (preinstalled here) — no trainer import. It merges
the training-time factorizations, applies the quantization scales, and re-validates
the archive (including int16 accumulator headroom).

## Verify (do this before every upload)

`tools/verify_export.py` runs in the **trainer** venv and checks that our integer
engine reproduces the trainer's own forward pass bit-for-bit (float64, quantization
epsilon disabled). A mismatch means the export misread the checkpoint.

```bash
TRAINER=~/aichessathon-train/nnue-pytorch \
PYTHONPATH=$TRAINER \
$TRAINER/.venv/bin/python tools/verify_export.py \
  $TRAINER/runs/net1/lightning_logs/version_0/checkpoints/last.ckpt \
  --trainer $TRAINER --positions 300
```

Then `make gate` and `make zip` from the starter repo as usual.
