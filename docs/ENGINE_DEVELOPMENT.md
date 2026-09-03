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
