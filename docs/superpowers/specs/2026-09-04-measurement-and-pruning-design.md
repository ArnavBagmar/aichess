# Phase 5: honest measurement and the rest of the pruning set

**Written** 2026-09-04, branch `nnue`. Follows `2026-09-04-next-steps.md`, which is the
handoff this spec turns into work.

## Problem

The agent searches at about 22 k nodes per second and reaches depth 5 to 7 in a 4 s move.
Stockfish at the same clock reaches depth 20 or more on one core. That gap, not the
evaluation, is why the agent loses games: the profile in the phase 4 spec puts evaluation
at 4% of search time and python-chess move generation at most of the rest.

Two things block progress on depth:

1. **Nothing can be measured.** The only ladder is Stockfish with `UCI_LimitStrength`,
   which at 120 s + 0.5 s plays the same strength at 1900, 2200 and 2500. A change worth
   30 Elo is invisible in a 10-game run against it.
2. **The pruning set is half built.** Null-move and LMR are in. PVS, aspiration windows,
   check extensions, reverse futility pruning, staged move generation and quiescence
   pruning are not, and each buys plies without buying nodes.

This spec covers building the measurement, then landing the pruning set through it, one
change per commit, each accepted or reverted on its own evidence.

## How Stockfish does it, and what transfers

Stockfish's strength is three things: bitboard move generation in C++ at millions of
nodes per second, a net trained on billions of engine-labelled positions, and a search
whose every heuristic was accepted only after a self-play SPRT on fishtest. The first is
out of reach without a rewrite, which is its own later spec. The second is being trained
on the GPU in the background. The third transfers directly, and it is the working method
of this phase: every search change is a hypothesis, and games decide it.

Re-implementing the published techniques from the Chess Programming Wiki is normal
engine work and allowed. Pasting Stockfish source is a rules violation and a GPL problem
and does not happen.

## Constraints

- One core, 2 GB, 120 s + 0.5 s wall-clock. A flag loses, so time control stays as built.
- `tools/` never ships: `harness/package.py` globs root-level `*.py` and `weights/`.
  Stockfish stays at `C:/Users/arnav/stockfish/`, outside the repo.
- `harness/` is not edited. The bench keeps going through `harness.referee.play_match`
  so the clock and the loss conditions stay the platform's.
- The machine has 6 physical cores. Every bench run keeps `workers * 2 <= 6`, because
  the referee measures wall time and an oversubscribed box makes the agent look weaker.
- The GPU is busy with training during this phase. Nothing here needs it.

## Part 1: measurement

All changes live in `tools/elo_bench.py`, which already wraps Stockfish as a
`harness.sandbox.Agent`, generates seeded near-equal openings, plays colour-reversed
pairs, and runs pairs concurrently.

### 1a. Self-play: `--opponent DIR`

Plays the agent in `--agent` against the agent in `--opponent`, both launched with
`harness.sandbox.local`, so each is the same one-core process the platform runs. The
opponent is normally a git worktree of the commit before the change under test:

```
git worktree add ../aichessathon-base HEAD~1
uv run python tools/elo_bench.py --opponent ../aichessathon-base --sprt --workers 3
```

`--opponent` and `--elo` / `--nodes` are mutually exclusive. `play_pair` takes a factory
for the opposing agent instead of Stockfish parameters, so the pair logic is shared.

The opponent worktree needs `weights/nnue.npz`, which is gitignored. The bench checks
for it and fails with a message naming the file rather than letting the opponent crash
at init and score as a loss.

### 1b. Sequential probability ratio test: `--sprt`

Replaces the fixed game count. Uses the generalised SPRT on trinomial results, as
cutechess-cli does:

```
s0, s1 = expected score at elo0, at elo1      s = 1 / (1 + 10 ** (-elo / 400))
LLR = games * (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance)
```

where `mean` and `variance` are the sample mean and variance of per-game scores
(1, 0.5, 0). The run stops when LLR crosses `log((1 - beta) / alpha)` (accept: the
change is at least `elo1` better) or `log(beta / (1 - alpha))` (reject: it is no better
than `elo0`). Defaults: `elo0 = 0`, `elo1 = 20`, `alpha = beta = 0.05`, so the bounds are
about ±2.94. Tight fishtest bounds like [0, 5] would take days at Python speed; a
depth-starved engine gains in large steps, so [0, 20] resolves in an hour or two.

LLR is updated once per completed pair under the existing lock, and the run ends after
the pair that crosses a bound. `--games` becomes a ceiling in SPRT mode, defaulting to
400, so a change that is exactly on the boundary cannot run forever. The report prints
the verdict, the LLR, the bounds, and the usual tally.

A `--sprt` run in the default configuration uses a 10 s base with the platform's real
0.5 s increment, because self-play is about the difference between two versions and the
short base resolves that difference in a fraction of the time. The increment is kept
because the search budgets for it: at the arena's 0.1 s it overspends by about 200 ms a
move and flags in long games (seen in the Task 2 smoke test), and flagged games are
noise. The phase 4 spec's warning stands: local scores at a short clock understate
absolute strength. They do not mis-order two versions of the same engine.

### 1c. Node-limited Stockfish: `--nodes N`

Replaces `--elo` for the Stockfish opponent. Full-strength Stockfish searches exactly
`N` nodes per move via `chess.engine.Limit(nodes=N)` and ignores the clock, so its
strength is monotonic in `N` and independent of time control. The ladder is 1k, 4k,
16k and 64k nodes. It is the absolute yardstick, run once per milestone, not per change.

`report` learns to describe the opponent as "Stockfish at N nodes" and stops printing an
estimated rating in that mode, since there is no Elo to anchor to. Score and the Elo
difference from that opponent are enough.

### Measurement acceptance

- `--opponent`, `--sprt` and `--nodes` each have a unit test that exercises the parsing
  and the arithmetic without launching a game. The SPRT test pins the LLR for a known
  tally against a hand-computed value and checks both stopping rules.
- A baseline self-play run, current `HEAD` against itself, must **not** cross either
  bound within 100 games. That is the sanity check that the harness has no colour or
  ordering bias before it judges anything.
- A 20-game run at `--nodes 4000` is recorded in the session log as the milestone
  baseline before any pruning change lands.

## Part 2: the pruning set

Each item is its own commit on `nnue`, in this order, and each is accepted only if its
`--sprt` run passes. A failed run is reverted, not tuned by eye. A change that passes
but slows nodes per second by more than 10% is reported as such in its commit message.

All of it lives in `search.py`. `nnue_engine.py`, `nnue_features.py` and the harness
stay untouched.

### 2.1 Transposition table mate scores

Correctness, not strength, so no SPRT: a test pins it instead. Scores within
`MAX_PLY_LIMIT` of `±MATE` are converted from "mate in N from the root" to "mate in N
from this node" on store, and back on probe, by adding or subtracting `ply`. Without
this, a mate read back at a different ply reports the wrong distance and the engine can
dither in a won position.

The same commit replaces `table.clear()` on overflow with depth-preferred replacement:
a new entry overwrites an existing one for the same key only if its depth is at least as
large or the existing entry is from an earlier search. Overflow of the dictionary itself
is handled by clearing once and noting it; a real two-tier table is deferred until the
profile says the table matters.

### 2.2 Principal variation search

After the first move at a node is searched with the full window, every later move is
searched with a null window `(alpha, alpha + 1)`. If it fails high and the window is
wider than one, it is re-searched with the full window. LMR's re-search rule folds into
this: a reduced move that beats alpha is re-searched at full depth with the null window
first, and only then with the full window. The root uses the same scheme.

Expected: the same tree with far fewer exact-window nodes, so more depth per second.

### 2.3 Aspiration windows

From depth 5 on, the root search starts with a window of ±50 cp (in internal units,
±1600) around the previous iteration's score. A fail-low or fail-high widens the window
on that side by doubling until it succeeds or the window is open. A fail-low is also
the signal that the previous best move is in trouble, and the existing abort rule stays:
a depth interrupted by the clock is discarded whole.

### 2.4 Check extensions

A move that gives check is searched one ply deeper. Bounded by `MAX_PLY_LIMIT` as
today, and never combined with a reduction, which LMR already guarantees by skipping
moves that give check. Expected: forcing sequences no longer stop at the horizon, so the
engine sees the mate threats it currently walks into.

### 2.5 Reverse futility pruning

At depth 1 to 3, not in check, and with `beta` not a mate score, if the static
evaluation minus a margin of `depth * 120 cp` is still at least `beta`, return the
static evaluation. This is the single cheapest cut in modern engines and the one that
most rewards an evaluation that is nearly free to call.

### 2.6 Staged move generation

`ordered_moves` today generates and sorts every legal move at every node, including the
majority that cut off on the TT move or the first capture. It becomes a generator that
yields in stages: the TT move if legal, then captures ordered by MVV-LVA, then killers,
then the remaining quiets ordered by history. The transposition key is computed once per
node and passed down instead of recomputed. `is_capture` is called once per move.

This is the only speed item in the set. It is accepted on nodes per second measured on
the four profile positions from the phase 4 spec, and confirmed by SPRT.

### 2.7 Quiescence pruning

Delta pruning: in quiescence, a capture whose victim value plus a 200 cp margin cannot
raise the static evaluation to `alpha` is skipped. Then a static exchange evaluation for
ordering captures and skipping losing ones. These are last because quiescence is a
smaller share of the tree than the items above, and because SEE on python-chess is not
cheap.

## Testing

- Every search change ships with a unit test in `tests/test_search.py` that exercises
  the new code path on a small position, alongside the existing 26 tests. Mate-distance,
  window widening and the staged generator's ordering are all pinnable without games.
- The existing tactical and endgame tests keep passing at every step; they are the
  guard against a pruning rule that is wrong rather than merely weak.
- `make gate` before every commit: ruff, mypy strict, and two clean games.

## Out of scope

- **numba bitboard move generation.** The largest lever available and a rewrite that
  moves `nnue_features` off python-chess boards. Its own spec, after this phase gives it
  a bench to be measured on.
- **The L1 512 net.** Queued for the GPU after the current resume run exports. Its own
  short spec: arch constant, training command, export, verify, SPRT against the L1 256
  net.
- **Pondering.** Still permitted and still valuable; still deferred behind cheaper items.
- **Time management tuning.** Not touched until the search is stable enough for it to
  matter.

## Session context, 2026-09-04

- The L1 256 net is resuming from epoch 108 toward 150 on the GTX 1660, launched at
  17:31 detached via `Start-Process`, logging to `D:/aichessathon-train/train2.log` and
  writing to `runs/net1/lightning_logs/version_2/`. Expected to finish in about 3.7 h;
  `--max-time 00:06:00:00` is the safety stop. When it ends: export with
  `tools/export_net.py`, verify with `tools/verify_export.py` in the trainer's venv, and
  SPRT the new net against the old one with the bench from Part 1.
