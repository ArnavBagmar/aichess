# Engine development and evidence plan

This document turns the next investigation areas into concrete work. It reflects the live
[competition documentation](https://aichessathon.com/docs) checked on 2 September 2026. The live
page remains authoritative: the submission is **50 MB unzipped**, and every rated game begins from
a curated position intended to be close to level.

## 1. Competition-game analysis

There are no public competition games yet. The
[leaderboard](https://aichessathon.com/leaderboard) says round one begins 4 September at 08:00,
and the [archive](https://aichessathon.com/archive) will publish match data after games exist.

Prepare an importer now. For every public game, retain:

- starting FEN, side to move, castling rights, en-passant square, material, and game phase;
- result, termination reason, total plies, and color assignment;
- move times or clock snapshots when published;
- flags, crashes, illegal moves, repetitions, and adjudications;
- evaluation swings and the first large mistake, when offline analysis is available;
- anonymized opponent archetype and engine-version identifier.

Produce distributions for starting material, phase, pawn structure, game length, termination, time
usage, and performance by color. Public-match analysis is permitted; hidden match data, another
participant's system, and organizer infrastructure are out of scope under the competition rules.

## 2. Training and test datasets

The [Lichess open database](https://database.lichess.org/) provides CC0 game exports, hundreds of
millions of Stockfish-evaluated positions, and millions of rated, tagged puzzles. Stream and sample
these sources; do not commit the bulk archives.

Create versioned manifests for four datasets:

### `quiet_eval`

Positions for static-evaluator training:

- legal, nonterminal, tactically stable positions;
- deep teacher value, depth, nodes, source, and optional game result;
- balanced by side to move, phase, material, and score range;
- deduplicated by normalized position key;
- split by source game or opening family to prevent leakage.

### `move_ranking`

Multi-PV positions containing each candidate move, teacher score, depth, and best/second-best gap.
Use these to test root ordering, pairwise ranking, or a tiny policy head.

### `tactics`

Lichess puzzles grouped by theme, rating, phase, and solution length. The stored puzzle FEN is before
the opponent's setup move: apply the first listed move, then ask the engine for the second move.
Track exact solution, acceptable equivalent mates, search time, depth, and nodes.

### `search_instability`

Positions where shallow and deep searches disagree in score, best move, or principal variation.
These are valuable for pruning regressions and adaptive time-management tests.

Additional targeted slices should cover endgames, fortresses, defensive only-moves, close root
choices, and positions with unresolved captures or checks.

## 3. Correctness and adversarial tests

Correctness precedes Elo tuning. Add automated coverage for:

- standard perft reference positions at affordable depths;
- legal castling and castling through check;
- en passant, including discovered rook or bishop attacks;
- promotion and underpromotion;
- single check, double check, pins, and check evasions;
- checkmate, stalemate, insufficient material, and terminal scoring;
- threefold history and the fifty-move counter;
- FEN round trips and unusual but legal FENs;
- push/pop restoration and incremental hash/evaluation restoration;
- TT key distinctions for side, castling, and en-passant state;
- exact/lower/upper TT bounds and depth replacement;
- mate-score ply normalization across TT storage and retrieval;
- null-move safeguards in check and zugzwang-prone endings;
- deadlines from 1 to 100 ms with an always-legal fallback;
- long games up to the 300-ply adjudication boundary;
- bounded memory across repeated searches and a whole game;
- randomized legal-game fuzzing with invariants after every push/pop.

Stockfish's [perft test](https://github.com/official-stockfish/Stockfish/blob/master/tests/perft.sh)
contains useful public reference positions. The public
[Chess-EPDs collection](https://github.com/ChrisWhittington/Chess-EPDs) contains further stress,
fortress, quiet, and tactical positions; check provenance and licensing before importing any suite.

## 4. Platform-faithful performance measurement

Maintain a deterministic corpus of 50-200 FENs spanning phases, material levels, checks, tactics,
quiet positions, and high/low branching factors. Benchmark on Linux with Python 3.12, one pinned CPU
core, the five allowed package versions, no network, and the competition memory limit.

Record per position and in aggregate:

- wall and CPU time, peak memory, and initialization time;
- legal moves/s, evaluations/s, search nodes/s, and quiescence nodes/s;
- completed depth and effective branching factor;
- TT probes, hits, useful cutoffs, replacements, and occupancy;
- beta cutoffs and first-move cutoff rate;
- null-move attempts/cutoffs and LMR reductions/re-searches;
- aspiration failures and principal-variation changes;
- time in move generation, push/pop, ordering, evaluation, and neural inference;
- evaluator incremental-update cost versus full recomputation.

Profile real searches before rewriting hot paths. `python-chess` is a strong correctness baseline,
but board copying, push/pop, legal-move objects, and sorting may become bottlenecks. Use Numba only
for stable numeric kernels that can remain in nopython mode, and include JIT warm-up in the 60-second
initialization budget.

## 5. Compact endgame knowledge

The rules permit books and tablebases as shipped data, and `chess.syzygy` is preinstalled. The
complete five-piece Syzygy set does not fit the 50 MB submission, so test small options rather than
assuming a full tablebase deployment:

- complete three-piece tables;
- four-piece WDL tables if their measured packaged footprint is worthwhile;
- selected frequent five-piece material classes;
- WDL-only files when DTZ does not justify its size;
- handcrafted king-and-pawn, opposition, passed-race, mop-up, and mating logic.

Test probe latency, table frequency on the curated-position corpus, search integration, fifty-move
behavior, packaged size, and match Elo. A tablebase cutoff must preserve correct WDL semantics and
must not confuse a cursed win with a fifty-move-rule win.

## 6. Opening and starting-position strategy

Every rated game starts from a curated near-equal position, so do not optimize primarily for move
one of normal chess. Until public games reveal repetition in the position pool:

- prefer general evaluation and search over a conventional opening book;
- train across early, middlegame, and endgame FENs;
- handle unfamiliar pawn structures and castling rights immediately;
- consider a tiny cache only if public starting positions repeat;
- test a root policy as a general move-ordering tool, not merely opening memorization.

Any book competes with evaluator weights, tables, and code for the 50 MB package budget.

## 7. Evaluation and WDL calibration

Measure more than mean squared centipawn error:

- best-move, top-k, and pairwise move-ranking accuracy;
- WDL calibration and expected calibration error;
- error by material, phase, score range, and tactical stability;
- shallow/deep search disagreement;
- search nodes saved through improved ordering or leaf accuracy;
- paired-game Elo at fast and competition time controls.

The [Stockfish WDL model](https://github.com/official-stockfish/WDL_model) uses logistic curves whose
parameters depend on remaining material. This captures that the same numerical advantage has
different winning chances in different endgames. Fit our own calibration from our engine's games;
do not assume Stockfish's centipawn scale transfers to our evaluator.

Keep all value conventions explicit: side-to-move perspective, mate range, centipawn clipping,
terminal values, and TT normalization. Validate them with symmetry and color-flip tests.

## 8. Opponent-robust internal league

Build legal, deterministic reference agents with deliberately different profiles:

- material-only greedy;
- shallow full-width minimax;
- positional search without quiescence;
- tactical search with weak positional evaluation;
- aggressively pruned search;
- compact learned evaluator with shallow verification;
- repetition-seeking or draw-preferring play;
- deliberately constrained time management;
- randomized legal play for robustness and fuzzing.

Use paired starting positions with colors reversed. Record W/D/L, pentanomial pair outcomes, flags,
crashes, illegal moves, latency, and version hashes. A candidate must beat the current version across
multiple opponent types rather than overfit one baseline.

## Just-in-time research backlog

We have enough general theory to build. The remaining research is triggered by a concrete engine
milestone so reading stays connected to an implementation or measurement.

### Before learned evaluation

Study and experimentally settle:

- king-relative sparse features and dual-perspective accumulators;
- eager versus lazy accumulator updates;
- refresh behavior after king moves and unusual state transitions;
- integer scale factors, clipping, saturation, and worst-case overflow bounds;
- quantization-aware training and agreement between training and deployed inference;
- phase/material output buckets and whether their gain pays for added complexity;
- NumPy, Numba, and ONNX latency at the batch size of one used by alpha-beta search.

The [official NNUE guide](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
is the primary implementation reference. We will use its mathematical principles but train our own
weights and write our own integration.

### Before each selective-search feature

Research the interactions and construct regression positions before enabling the feature:

- LMR with history scores, checks, promotions, and tactical re-searches;
- null-move pruning with check, low material, zugzwang, and verification search;
- aspiration widening under unstable scores;
- fail-soft bound semantics and TT storage after reduced/null-window searches;
- futility, reverse futility, razoring, and delta pruning near mate or tactical scores;
- extension stacking and pathological search explosion.

Never introduce several pruning mechanisms in one experiment. Correctness and paired-game evidence
are required because an apparently faster search can become tactically unsound.

### When profiling identifies a Python bottleneck

Benchmark alternatives for the measured hot path rather than assuming a full rewrite is necessary:

- `python-chess` legal generation, `push()`, `pop()`, check detection, and move-object allocation;
- fixed NumPy arrays versus dictionaries for the transposition table;
- compact integer moves versus `chess.Move` instances inside ordering tables;
- public or team-written position hashes versus private library internals;
- garbage-collection behavior and allocation-free search stacks;
- JIT boundary overhead and the amount of work required per Numba call.

Retain python-chess as the legality oracle in tests even if a faster internal representation is
eventually introduced.

### Before adaptive time management

Measure a conservative base allocator first, then research bonuses based on:

- principal-variation and best-move stability;
- root-score oscillation and best/second-best separation;
- aspiration failures, branching factor, and only-move detection;
- increment, remaining time, estimated moves remaining, and fixed move overhead;
- ponder hits and safe cancellation of obsolete pondering work.

Stockfish describes a base allocation modified by best-move, evaluation, and position-complexity
stability. Its constants are not portable; fit ours from our own telemetry at 120+0.5.

### Before trusting Elo results

Implement paired/pentanomial reporting, confidence intervals, and sequential stopping. Determine
the minimum Elo effect our available game count can detect. Track fast-control versus 120+0.5
correlation and avoid selecting the best-looking result from many noisy experiments.

### After public ladder games appear

Replace assumptions with observed distributions: starting FEN repetition, phase, material,
practical asymmetry, game length, termination, and opponent failure modes. Engine-near-zero does not
necessarily imply equal practical difficulty, so always reverse colors in local tests when possible.

### Continuous failure mining

Automatically retain positions where the chosen move loses teacher value, added depth changes the
best move, pruning changes the result, evaluation and search disagree, time usage spikes, or a match
loss begins. Minimize and categorize these positions, then add them to a versioned private regression
suite. Our own failures should eventually be more valuable than generic puzzles.

### Before spending package space on endgames

Measure how often each material class occurs, exact WDL/DTZ footprint, probe latency, fifty-move
semantics, and paired-game benefit. Compare every tablebase subset against using the same 50 MB
budget for evaluator weights or other frequently used data.

## Delivery order

1. Correctness, rule-edge-case, and deadline tests.
2. Deterministic benchmark corpus and search telemetry.
3. Public competition-game importer ready before round one.
4. Small, reproducible Lichess tactical sample and manifest.
5. Iterative alpha-beta baseline with quiescence and time safety.
6. Diverse internal opponent league and paired-match reporting.
7. Search improvements one at a time.
8. Compact trained evaluation, WDL calibration, and selective tablebases after profiling.

Every change should connect to a test, benchmark, dataset version, or paired match. Ideas that cannot
be measured should not displace work that can.

## Measured implementation checkpoints

- The exact Numba material/PST kernel measured 157,587 evaluations/s versus 30,710/s for the
  Python loop, including bitboard encoding (5.13x).
- Integrated whole-engine throughput increased from 16,122 to 21,026 nodes/s (30.4%) on the
  ten-position corpus, with extra completed depth in tactical positions.
- In 20 reversed-color games from real Lichess positions, the compiled evaluator scored
  `+8 =6 -6` against the exact Python checkpoint (55%, +34.9 unanchored Elo), with no failures.
- The realistic source sample came from the official CC0 Lichess January 2013 standard archive:
  100,000 games scanned, both ratings at least 1800, 5,000 deterministic reservoir positions.
- Analysis of seven draws against Stockfish 1320 found six began worse for our agent; repetition,
  stalemate, and fifty-move handling saved points rather than wasting wins. The sole favorable start
  later crossed a low-clock depth threshold: at 1--2.5 seconds the engine selected `e4f5` at depth 1,
  while a slightly larger emergency allocation completed depth 2 and selected `f4e5`. The revised
  policy scored `+8 =4 -8` against the previous policy over 20 reversed-color games with no flags.
- The first fixed five-level Stockfish 18 ladder (five real-position pairs per level, 3+0.1, 100 ms
  teacher moves) scored: 85% at 1320, 40% at 1500, 40% at 1700, 30% at 1900, and 25% at 2000.
  These ten-game levels have wide uncertainty, but the 2000 score is about -191 unanchored Elo and
  establishes the gap. On two real-position pairs apiece, the engine scored 4-0 against every
  bundled baseline: random, greedy, minimax, Numba, and approximate-1000.
- Profiling showed eager `gives_check()` calls consumed roughly 14% of search time. Restricting
  them to actual LMR candidates and otherwise-prunable quiescence moves increased throughput from
  21,132 to 23,187 nodes/s (9.7%) without changing search semantics. The optimized build scored
  `+9 =5 -6` against its checkpoint over 20 real-position games (+52.5 unanchored Elo), with no
  failures.
