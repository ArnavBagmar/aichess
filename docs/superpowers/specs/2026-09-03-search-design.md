# Phase 3: search

## Problem

`agent.py` picks moves by evaluating every legal move one ply deep and taking the best
static score. A one-ply agent hangs material to any two-move tactic no matter how good
its evaluation is, so the NNUE we trained is largely wasted. `docs/IDEAS.md` is direct
about where the strength actually comes from: "Negamax with alpha-beta is the whole
game."

This spec covers replacing that one-ply loop with a real search.

## Constraints

From the agent contract, which the design has to respect rather than discover later:

- One core, 2 GB, wall-clock time control of 120 s + 0.5 s per move. A flag loses.
- `get_move(fen, time_left_ms) -> str` is the only input. There is no move history,
  no opponent model, no engine handle.
- The process lives for one game and dies. Module state survives between our moves in
  the same game and never reaches the next game.
- 60 s import budget, before the clock starts. numba compilation belongs there.
- Illegal move, malformed output, crash, or OOM loses the game outright.

## Architecture

One new module, `search.py`, holding a single `Searcher` class. `agent.py` keeps its
current shape — module-level construction at import, a thin `get_move` with a
fallback — and delegates the choice to a module-level `Searcher`.

```
agent.py          get_move(fen, time_left_ms) -> str
   |                 constructs once at import, catches everything
   v
search.py         Searcher.pick(fen, time_left_ms) -> chess.Move
   |                 owns: TT, killers, history, clock, game history
   v
nnue_engine.py    Engine.push / pop / evaluate       (unchanged)
   v
nnue_features.py  incremental accumulator indices    (unchanged)
```

`Searcher` drives the existing `Engine` through `push`/`pop`, so the incremental
accumulators keep behaving exactly as `tests/test_parity.py` already pins them. No
changes to `nnue_engine.py`, `nnue_features.py`, `nnue_net.py`, or `nnue_arch.py`.

### Why a class rather than module globals

The search needs five pieces of mutable state that must not leak between games and
must be inspectable from tests: the transposition table, killer moves, the history
table, the deadline, and the game's position history. A class makes each test able to
construct a fresh, isolated searcher; module globals would make test order matter.

## Units and score conventions

`Engine.evaluate()` returns an integer in 1/32-centipawn units, positive for the side
to move. The search stays in those units end to end — no float conversion, and no
second scale to keep in sync with `nnue_arch`.

- `MATE = 1_000_000`, matching the constant `agent.py` already uses.
- A mate found at distance `ply` scores `MATE - ply`, so shorter mates are preferred
  and mate scores stay comparable across depths.
- Draws — stalemate, insufficient material, 50-move, repetition — score exactly `0`.

## Components

### Iterative deepening

`pick` searches depth 1, then 2, then 3, and so on until the budget is gone. Each
completed depth overwrites the best move. Depth 1 completes in well under a
millisecond, so a legal move is always in hand before any deep search starts — this is
what makes the time control safe rather than hopeful.

### Negamax with alpha-beta

Standard fail-soft negamax. At each node:

1. Return a draw score for repetition, 50-move, or insufficient material.
2. Probe the TT; use the stored score on a sufficient-depth hit with a usable bound,
   and otherwise keep the stored move for ordering.
3. At depth 0, drop into quiescence.
4. Generate and order moves, recurse, update alpha, cut on beta.
5. Store the result in the TT.

No legal moves means checkmate (`-MATE + ply`) if in check, stalemate (`0`) otherwise.

### Quiescence

At the leaves, search captures and queen promotions only, with a stand-pat cut on the
static evaluation. Without it the evaluation is measured mid-exchange and is simply
wrong, which `docs/IDEAS.md` calls out explicitly.

### Move ordering

Alpha-beta only pays when good moves come first, so ordering is the highest-leverage
part of this design:

1. The TT move, if any.
2. Captures and promotions by MVV-LVA.
3. Two killer moves per ply.
4. Remaining quiet moves by a history table keyed on (from-square, to-square).

### Transposition table

A plain dict keyed on `board._transposition_key()`, which `docs/IDEAS.md` suggests
directly. Each entry stores depth, score, bound type (exact / lower / upper), and the
best move.

The table persists across our moves within a game, which is a real gain since the
opponent usually plays into the tree we just searched. It is capped at `TT_MAX_ENTRIES
= 200_000` and cleared wholesale on overflow — crude, but it bounds the table to
roughly a hundred megabytes, well inside 2 GB, and costs nothing on the hot path. The
process is per-game, so no cross-game contamination is possible.

### Time management

Budget for a move is derived from the clock we were handed, never from a constant:

```
MOVES_REMAINING = 30      # assumed horizon, self-correcting as the clock changes
INCREMENT_MS    = 500     # published time control: 120 s + 0.5 s per move
SAFETY_MS       = 50      # wall-clock margin the referee does not forgive
MAX_FRACTION    = 0.4     # never spend more than this much of what is left
MIN_BUDGET_MS   = 10      # always attempt something

budget = time_left_ms / MOVES_REMAINING + 0.6 * INCREMENT_MS
budget = min(budget, MAX_FRACTION * time_left_ms)
budget = max(budget - SAFETY_MS, MIN_BUDGET_MS)
```

`INCREMENT_MS` is the one number taken from the published rules rather than the input,
so only a fraction of it is claimed; if the platform ever changed the increment, the
result is a slightly conservative budget rather than a flag.

The deadline is checked inside the search every `CLOCK_CHECK_NODES = 2048` nodes, not
only between iterations. Exceeding it raises `SearchAborted`, which unwinds to the
iterative-deepening loop. That loop discards the incomplete depth entirely — a
half-searched depth has a biased best move — and returns the best move from the last
completed depth.

The referee measures wall time and does not forgive, so the reserve is deliberate
margin, not an estimate.

### Cross-move state: repetition

Each call appends the current position's key to a game-history list. Inside the
search, a position matching one already in that history, or one already on the current
search path, scores as a draw.

This is worth real points in both directions: it lets us claim a repetition when we
are losing, and steer away from one when we are winning. The referee claims threefold
automatically, so a won game can otherwise be drawn without us ever being told.

We only observe positions where it is our turn — every other ply — which is enough,
since a position repeating at our turn is a genuine repetition of that position.

A position that cannot follow the one before it — detected by its fullmove number
going backwards — resets the history rather than carrying a wrong assumption forward.
Within a single game this should never fire; it exists so a surprise cannot turn into
a bogus draw claim.

## Error handling

`get_move` keeps its current structure: the entire search is wrapped, and any
exception falls back to a legal move rather than losing the game. A bug in our search
should cost strength, not the point. `print` is safe — the runner redirects fd 1 away
from the protocol stream.

The searcher itself raises rather than guesses on genuinely impossible input (a FEN
with no legal moves), and `get_move` converts that into the fallback.

## Warm-up

`agent.py` already calls `nnue_engine.warm_up` at import to compile the numba kernels
inside the 60 s budget. That is extended to run a short shallow search, so the search's
own code paths and any argument-type specializations are compiled before the clock
starts rather than on our first move.

## Testing

New `tests/test_search.py`:

- Finds a mate in one, and a mate in two, from known positions.
- Never returns an illegal move, across randomized positions including promotions and
  en passant.
- Respects a tight time budget — returns within the deadline plus a small tolerance.
- Scores a forced repetition as a draw.
- Handles the no-legal-moves and single-legal-move cases.
- A fresh `Searcher` is isolated: no state leaks between instances.

The existing gate stays authoritative: `ruff`, `mypy --strict`, and arena games that
have to finish cleanly.

## Measuring

Two games tell you nothing. The comparison that matters is against our own previous
version, so the searching agent is measured against the current one-ply agent over
enough games at a fast time control for the score to mean something, colours
alternating. `make arena` against the `greedy` and `minimax` baselines is the
secondary check.

## Non-goals

Deliberately excluded from this pass, to be revisited with arena data in hand:

- **Pondering.** Allowed by the rules, and the process does keep its core while the
  opponent thinks, but it adds real failure modes for a speculative gain.
- **Null-move pruning and late-move reductions.** The largest remaining depth
  multipliers, but both need a working, measured baseline first.
- **Aspiration windows.**
- **A numba bitboard move generator.** The single biggest speed lever, and the reason
  node counts here are thousands rather than millions. It would also force rewriting
  `nnue_features` off python-chess boards, so it is a separate project with its own
  risk of illegal-move bugs that lose games outright.
- **An opening book.** Rated games start from curated positions, so a book keyed on
  move one is often already out of book.
