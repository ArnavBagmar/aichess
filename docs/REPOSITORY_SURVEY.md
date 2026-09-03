# Chess-engine repository survey

This is a clean-room architecture survey of public chess projects inspected for the AI Chessathon.
It identifies reusable ideas and testing practices, not code to copy. The competition prohibits
shipping or wrapping an existing engine, and submitted learned models must be trained by the team.

## Repositories inspected

### Stockfish

- Repository: [official-stockfish/Stockfish](https://github.com/official-stockfish/Stockfish)
- Study value: the reference architecture for a modern alpha-beta engine.
- Concepts: iterative deepening, aspiration windows, principal-variation search, transposition
  bounds, move-ordering tables, null-move pruning, late-move reductions, extensions, quiescence,
  NNUE, WDL reporting, and time management.
- For us: independently implement the general algorithms in a simpler Python design; do not port
  Stockfish source or bundle its network.

### nnue-pytorch

- Repository: [official-stockfish/nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
- Study value: sparse features, data preparation, quantization, serialization, and the separation
  between offline training and fast engine inference.
- For us: begin with a small feature set, benchmark one-core latency, and train only networks cheap
  enough to call throughout search.

### Fishtest

- Repository: [official-stockfish/fishtest](https://github.com/official-stockfish/fishtest)
- Study value: paired openings, color reversal, pentanomial outcomes, sequential tests, normalized
  Elo, and SPSA tuning.
- For us: reproduce these principles locally and never judge strength from a handful of games.

### fastchess

- Repository: [Disservin/fastchess](https://github.com/Disservin/fastchess)
- Study value: clocks, engine-process orchestration, opening repetition, adjudication, result
  accounting, and PGN output.
- For us: harden the supplied harness and retain enough metadata to reproduce every experiment.

### Leela Chess Zero

- Repository: [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0)
- Study value: policy/value networks, PUCT-style search, batched inference, and self-play.
- For us: learned policy and value are useful ideas, but the usual GPU-oriented architecture is
  mismatched to one CPU core and a 50 MB submission.

### Ethereal

- Repository: [AndyGrant/Ethereal](https://github.com/AndyGrant/Ethereal)
- Study value: a readable strong-engine organization covering alpha-beta search, evaluation,
  tuning, and protocol integration.
- For us: coherent search/evaluation engineering matters more than any isolated pruning trick.

### Berserk

- Repository: [jhonnold/berserk](https://github.com/jhonnold/berserk)
- Study value: a compact modern alpha-beta/NNUE engine and the boundaries between board state,
  search heuristics, transposition storage, and neural evaluation.
- For us: keep hot paths data-oriented; Python object allocation in the tree must be profiled.

### Weiss

- Repository: [TerjeKir/weiss](https://github.com/TerjeKir/weiss)
- Study value: a smaller conventional engine whose search and evaluation organization is easier to
  trace than a mature reference engine.
- For us: keep the baseline understandable and heuristics isolated so they can be ablated.

### Sunfish

- Repository: [thomasahle/sunfish](https://github.com/thomasahle/sunfish)
- Study value: a minimal Python engine illustrating compact representation and high-level-language
  performance tradeoffs.
- For us: Python is viable, but competitive strength requires profiling, bounded structures, and
  possibly Numba for stable numeric kernels.

### Bullet

- Repository: [jw1912/bullet](https://github.com/jw1912/bullet)
- Study value: efficient NNUE training, features, quantization, and trainer throughput.
- For us: borrow offline methods while shipping only allowed dependencies and team-produced weights.

## Cross-project conclusion

Strong projects converge on this dependency chain:

```text
correct move generation and clocks
  -> iterative alpha-beta framework
  -> move ordering and transposition reuse
  -> selective reductions and pruning
  -> fast, calibrated leaf evaluation
  -> controlled match testing
```

A stronger evaluator cannot repair illegal moves or flag falls. Advanced pruning cannot help if
ordering searches poor branches first. An excellent search also cannot compensate indefinitely for
a slow evaluator on one core.

## Clean-room implementation rules

1. Record the paper or algorithm motivating a feature.
2. Write our own implementation against our own interfaces.
3. Do not paste surveyed source or translate it line by line.
4. Do not ship third-party engines, binaries, or pretrained engine weights.
5. Preserve licenses and attribution for any permitted dependency or data asset.
6. Validate every packaged file and expanded size before upload.
7. Treat current competition documentation and organizer rulings as final.

## Immediate implications

- Use a bounded transposition table rather than an unbounded Python dictionary.
- Avoid rebuilding expensive features at every node when incremental state is practical.
- Keep a legal move ready before beginning deeper iterations.
- Separate correctness, tactical, performance, and Elo testing.
- Treat time forfeits and crashes as first-class regressions.
- Version team-trained weights with dataset provenance and training configuration.
- Spend package size only on assets that measurably improve arbitrary-FEN strength.
