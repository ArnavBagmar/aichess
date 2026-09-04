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

## One-time setup (Windows + NVIDIA)

The macOS recipe above does not port directly. What works:

```bash
# The trainer venv needs a CUDA torch. requirements.txt has no torch pin, so it
# arrives CPU-only as a transitive dependency of torchmetrics; --reinstall is
# required because uv otherwise treats the requirement as already satisfied.
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe --reinstall torch \
  --index-url https://download.pytorch.org/whl/cu128

# cmake/ninja without a Visual Studio C++ workload:
uv tool install cmake && uv tool install ninja
```

Build the data loader with GCC, not MSVC: `data_loader/cpp/CMakeLists.txt` sets
`-march=native -Wall -O3` unconditionally, which `cl.exe` does not understand. MSYS2
mingw-w64 g++ works, with three fixes:

1. `lib/parallel_dataloader.h` guards `gmtime_s` behind `#if defined(_MSC_VER)`.
   MinGW is not MSVC but also has no `gmtime_r`; widen that guard to `_WIN32`.
2. Link the GCC runtime statically, otherwise `ctypes` cannot resolve
   `libstdc++-6.dll` / `libgcc_s_seh-1.dll` when it loads the DLL.
3. Copy `libwinpthread-1.dll` next to the built DLL. The static-pthread flags do not
   reach the link line through `Threads::Threads`, so that one stays a dynamic import.

```bash
cmake -S data_loader/cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/c/msys64/mingw64/bin/g++.exe \
  -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"
cmake --build build -j 12
cp /c/msys64/mingw64/bin/libwinpthread-1.dll build/
```

The `training_data_loader_bench` target fails to link under MinGW (duplicate
`_Unwind_Resume`). It is not needed: `libtraining_data_loader.dll` is the only
artifact `data_loader/_native.py` loads, and it globs `./build/**`, so `train.py` must
be run from the trainer root.

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

On CUDA, `pin_memory` and several workers are both fine, but the default
`--compile-backend inductor` needs Triton, which has no Windows build. Use
`--compile-backend cudagraphs`, which the trainer documents as the better fit for
small nets anyway:

```bash
.venv/Scripts/python.exe train.py <data1.binpack> <data2.binpack> \
  --features "HalfKAv2_hm^" --l1 256 \
  --accelerator cuda --gpus 0 --num-workers 4 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 150 --gamma 0.9788 \
  --default-root-dir runs/net1 --network-save-period 5 --save-top-k 8 \
  --compile-backend cudagraphs
```

Gotcha that costs real strength: the LR schedule is `StepLR(step_size=1,
gamma=0.992)`, stepped **per epoch and independent of `--max-epochs`**. That default
is sized for the Stockfish recipe of 400 epochs, where it anneals the LR to ~4% of
its initial value. Stopping a 400-epoch schedule early leaves the net at a high LR and
undertrained, so scale gamma to the run you can actually afford:
`gamma = 0.04 ** (1 / max_epochs)` (150 epochs -> 0.9788).

Past ~4 data-loader workers the GTX-class GPU is the bottleneck, not the loader; 4 and
8 workers measured identically (~313k positions/s at `--batch-size 16384`), so an
epoch of 100M positions takes about 5.5 minutes.

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
