# Phase 4 spec: where the agent stands and what to do next

**Written** 2026-09-04, at the end of the session that exported the trained net, built the
search, and measured the result. Branch `nnue`.

This document is the handoff. A new session should be able to read only this and be
productive.

---

## 1. Current state

The agent is complete and uploadable. `submission.zip` is 7.1 MB unzipped against a 50 MB
cap and contains `agent.py`, `nnue_arch.py`, `nnue_engine.py`, `nnue_features.py`,
`nnue_net.py`, `search.py`, `weights/nnue.npz`.

- **Net.** HalfKAv2_hm^, L1=256, trained to epoch 108/150 (~10.8B positions; the run hit
  its own 14 h limit early because the machine slept overnight). Train and val loss both
  0.00483, so it did not overfit. Exported from `last.ckpt` and verified **bit-exact**
  against the trainer's own forward pass: 300/300 positions, max diff 0.00 cp.
- **Search.** `search.py` holds a `Searcher` class: iterative deepening, negamax,
  transposition table, quiescence, MVV-LVA + killers + history ordering, budget-based time
  control, cross-move repetition history, **null-move pruning** and **late move
  reductions**. `agent.py` stays thin and delegates to it.
- **Tests.** 57 passing. `ruff` and `mypy --strict` clean across 26 files.

## 2. Measured baselines (all 2026-09-04, one core per side)

| Measurement | Result |
| --- | --- |
| greedy baseline, 40 games | **100%** (+40 =0 -0) — was 55% before the search |
| minimax baseline, 40 games | **98.8%** (+39 =1 -0) |
| Stockfish `UCI_Elo` 2200, 40 games, 120 s + 0.5 s | +30 =5 -5, **81.2%** |
| Stockfish `UCI_Elo` 2500, 10 games, 120 s + 0.5 s | +6 =3 -1, **75.0%** |

Across 90 tournament-rules games: **zero** crashes, illegal moves, flag falls, or ply-cap
adjudications. Time management holds under the real clock.

### The rating number is not trustworthy — read this before quoting one

We score roughly 75-81% against Stockfish set to 1900, 2200 **and** 2500. That cannot be
true if those settings differ by 600 Elo. `UCI_LimitStrength` weakens Stockfish by capping
its search rather than scaling with the clock, so at 120 s + 0.5 s the cap binds long
before the time does and every setting plays at roughly the same strength. The dial is
calibrated for much faster games than we play.

**Do not quote "2455 Elo" or "2691 Elo".** They are artefacts of a broken ladder.

**Fix this before optimising anything:** benchmark against **full-strength Stockfish capped
by nodes per move** (1k / 4k / 16k / 64k). Node-limited strength is monotonic and does not
collapse at long time controls. Test every change with **SPRT**, not eyeballed 10-game runs
— a 30 Elo gain is invisible in 10 games.

## 3. The profile that should drive every decision

Measured at a tournament budget (4.25 s per move) after null-move and LMR landed:

| position | depth | nodes | knps |
| --- | ---: | ---: | ---: |
| opening | 7 | 90,112 | 20.9 |
| middlegame | 5 | 86,016 | 19.8 |
| tactical | 6 | 90,112 | 20.8 |
| endgame | 10 | 94,208 | 21.8 |

cProfile of one 4 s middlegame search (6.8M calls):

- **python-chess board and movegen dominates** — hundreds of thousands of calls into
  `chess/__init__.py`
- `nnue_engine.push` 1.52 s cumulative (~38%), of which `nnue_features.move_deltas` 0.77 s
- **`Engine.evaluate` is 0.168 s — about 4%**

**The consequence that matters: evaluation is nearly free; move generation is not.**
A bigger net costs almost no search speed, so net quality is cheap. Making the search
faster means attacking python-chess, never the NNUE.

## 4. Ranked plan

Ordered by Elo per unit of effort.

1. **Finish the search-pruning set.** Null-move and LMR are in; still missing **PVS**,
   **aspiration windows**, **check extensions**, **SEE-ordered captures**, and
   **delta/futility pruning in quiescence**. At depth 5 the engine is still depth-starved,
   so plies bought without extra nodes remain the best value in the codebase.
2. **Transposition table correctness.** Mate scores are stored without adjusting for ply
   distance, so a mate read back at a different ply reports the wrong distance — it can
   make the engine dither in a won position. Add and subtract `ply` on store and probe.
   Also replace the crude `table.clear()` on overflow with a real replacement scheme
   (depth-preferred or two-tier).
3. **Cut python-chess overhead.** `_transposition_key()` is computed twice per node;
   `_move_score` calls `board.is_capture()` for every move; `ordered_moves` generates *and
   sorts* every legal move even at nodes that cut off on the first one. **Staged move
   generation** (TT move, then captures, then quiets) is the best speed-to-effort item here.
4. **numba bitboard movegen.** The 10-50x lever and the largest single win available, but a
   genuine rewrite: `nnue_features` reads python-chess boards and would have to move with
   it. Deliberately deferred through phases 3 and 4.
5. **The net.** Finish the remaining ~40 epochs (resume from `last.ckpt`; the epoch counter
   and LR schedule survive). Then train **larger — L1 512** — since evaluation is only 4%
   of search time. More engine-annotated training data is explicitly allowed.
6. **Pondering.** Explicitly permitted: the process keeps its core while the opponent
   thinks, so this roughly doubles usable thinking time. High value, but complex and easy
   to get wrong; do it after the cheap items.

Deliberate non-goals: an opening book (rated games start from curated positions, so it buys
little) and syzygy tablebases (the 50 MB cap allows only tiny ones).

## 5. Rules boundaries

- **Engine data for training is allowed.** `CLAUDE.md`: "Training on data an engine
  annotated is allowed; the ban covers what ships and runs inside the zip."
- **Re-implementing published techniques is fine.** Null-move, LMR, PVS and SEE are Chess
  Programming Wiki staples, not Stockfish property.
- **Never paste Stockfish source.** That is both a rules violation (checked after the fact,
  instant disqualification) and a GPL problem.
- **Nothing engine-derived ships.** Stockfish lives at `C:/Users/arnav/stockfish/`, outside
  the repo, and `tools/elo_bench.py` sits in `tools/`, which `harness/package.py` never
  packages — it globs root-level `*.py` plus `weights/`. Verified against the built zip.
- **Do not edit `harness/`.** It mirrors the platform protocol and clock.

## 6. Known warts

- **TT mate scores** are not ply-adjusted (item 2 above).
- **LMR trades exactness in won positions.** Where many moves win, it may choose a slower
  win over the fastest. `test_keeps_the_win_in_a_pawn_endgame` pins that the win survives
  rather than demanding one exact move. Accepted: it cannot hang material, and material
  adjudication at 300 plies would still score the win.
- **The local arena clock is not the platform's.** `harness/arena.py` defaults to
  10 s + 0.1 s while `search.INCREMENT_MS` is the real 500 ms, so local games are more
  time-pressured than rated ones and understate strength.
- **`tools/elo_bench.py --workers N`** must keep `N*2` within the **physical** core count
  (6 here, so 3). The referee measures wall time, so oversubscription makes the agent play
  weaker and understates the rating. Verified honest at 3 workers: ~7% per-pair slowdown.

## 7. Session log, 2026-09-04

1. Exported `last.ckpt` to `weights/nnue.npz`; `tools/verify_export.py` confirmed bit-exact.
2. Fixed the phase-2 export toolchain, which had landed with 8 ruff and 8 mypy failures.
3. Built the search from `docs/superpowers/plans/2026-09-03-search.md`, all five tasks TDD.
   The plan's in-check test carried an illegal FEN (`7r/8/8/8/8/8/6PP/6KR b`, no black
   king); replaced with `7k/8/8/8/8/8/6P1/6KR b`.
4. Measured both baselines to saturation (100% and 98.8%).
5. Built `tools/elo_bench.py` on top of `harness.referee.play_match`, later adding
   `--workers` after noticing one game leaves the machine ~85% idle.
6. Ran 40 games against Stockfish 2200 and 10 against 2500, which exposed the
   `UCI_LimitStrength` problem described in section 2.
7. Profiled the search, then added null-move pruning and LMR: +1 ply in
   opening/middlegame/tactical, +3 in the endgame.

## 8. Start here next session

In order:

1. Rebuild the Stockfish ladder on **fixed nodes per move** and wire up SPRT. Without it
   every later change is unmeasurable.
2. Fix the TT mate-score adjustment — a correctness bug, and cheap.
3. Add PVS, then aspiration windows, then check extensions, measuring each with SPRT.
4. Staged move generation.

Related memories: `improvement-roadmap`, `elo-benchmark`, `search-phase-3-plan`,
`nnue-training-run`.
