# Gators: an NNUE chess engine in pure Python

An entry for [AI Chessathon](https://aichessathon.com), built from scratch under the
competition's limits: one CPU core, 2 GB, no network, no GPU, no existing engine anywhere
in the zip, and Python source only. The engine is a HalfKAv2_hm NNUE evaluation I trained
on a GTX 1660 with the open [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
trainer, integer inference that reproduces the trainer bit-for-bit, and an alpha-beta
search compiled by numba over bitboards: about 300,000 nodes a second on one core, completing depth 13 to 17 in a
3.7 second middlegame move.

- **Write-up:** [docs/paper/index.html](docs/paper/index.html) covers the training runs,
  what I learned from the Stockfish source without shipping any of it, the search work,
  the node arithmetic, and the measurement discipline behind every change.
- **Play it:** `demo/` is a small web app that lets anyone play either of the two shipped
  networks and watch the search's depth, node counts, effective branching factor and
  principal variation move by move. See [Play it](#play-it).
- **Results:** 90% over the first ten rated games of the "Gators" build; about +200 Elo of
  gated self-play improvement on top of that in the builds that followed.

## What is in the repository

```
agent.py             the submission entrypoint: get_move(fen, time_left_ms) -> uci
search.py            iterative deepening, aspiration windows, time budgets (Python root)
search_kernel.py     the alpha-beta + quiescence search as one numba kernel
bitboard.py          bitboards, move generation, copy-make, Zobrist keys (numba)
nnue_bitboard.py     NNUE accumulators and the integer forward pass (numba)
nnue_arch.py         the network's shapes and quantisation constants
nnue_net.py          weight loading and validation
weights/nnue.npz     net2, the network the rated "Gators" games were played with
weights/nnue-net5.npz  net5, the latest network (~ +20 Elo over net2 on the same code)
tools/               export/verify a trained checkpoint, the SPRT bench, an nps probe
tests/               perft, differential and parity tests against python-chess, search tests
harness/             the competition's local referee, sandbox and packager (not edited)
demo/                the playable web demo (FastAPI + a vanilla-JS board)
docs/paper/          the write-up
docs/superpowers/    the design specs and implementation plans each phase was built from
```

## Running it locally

```
uv sync                      # Python 3.12; installs torch-cpu, numba, python-chess, the demo deps
make play                    # one game against a baseline at the real 120 s + 0.5 s clock
make arena                   # 20 fast games against a baseline
make gate                    # ruff, mypy --strict, and two games that must finish cleanly
uv run pytest                # perft, parity, export and search tests
make zip                     # submission.zip with agent.py at the root
uv run python tools/nps.py   # nodes per second on four fixed positions
```

Import takes 20 to 30 seconds: every numba kernel compiles at import so the compilation
lands in the competition's init budget rather than on the clock.

## Play it

```
uv sync
uv run uvicorn demo.app:app --port 7860
```

Then open <http://127.0.0.1:7860>. Pick a network, a colour and a think time; after every
engine move the panel shows the depth reached, nodes searched, nodes per second, the score,
the principal variation, and the per-depth node counts whose ratios are the effective
branching factor the search actually experienced.

The same app deploys to a Hugging Face Space (Docker, free CPU tier) with
`uv run python tools/deploy_space.py --repo-id <user>/aichessathon-engine` after
`hf auth login`. The Space's Dockerfile and card are in `demo/`.

## Training a network

`tools/TRAINING.md` is the full recipe: the trainer setup on Windows with a MinGW-built
data loader, the flags that match `nnue_arch.py`, exporting a checkpoint to `.npz`, and
verifying the export bit-for-bit against the trainer's own forward pass. Training data
was the public `test80`/`test77` binpacks; the rules allow training on engine-labelled
data, and nothing from the trainer ships.

## How changes were accepted

Every search change was one commit, played against a worktree of the previous commit in
self-play through the harness's real referee, and accepted or rejected by a sequential
probability ratio test (`tools/elo_bench.py --opponent ... --sprt`). The write-up's gate
table lists each result, including the one change that would have cost 141 Elo.

## Rules kept

No engine runs inside the submission: Stockfish was a sparring partner in the benchmark
only and lives outside the repository. Search techniques were re-implemented from the
Chess Programming Wiki; no Stockfish source was copied. `harness/` is the competition's
and is unmodified.

## License

MIT, as the starter it was forked from. The chess piece set in the demo is by Colin M.L.
Burnett, CC BY-SA 3.0.
