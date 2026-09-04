# Phase 5 Measurement and Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the bench a self-play SPRT mode and a node-limited Stockfish ladder, then land the remaining pruning techniques in `search.py` one SPRT-gated commit at a time.

**Architecture:** Part 1 (Tasks 1-4) extends `tools/elo_bench.py`, which already plays colour-reversed pairs through `harness.referee.play_match`: SPRT arithmetic, a `--nodes` cap for Stockfish, an `--opponent DIR` self-play mode, and a scheduler that stops once a verdict lands. Part 2 (Tasks 5-12) changes only `search.py`. Every strength change is a separate commit, tested by unit tests for the mechanism and by an SPRT run against a worktree of the previous commit for the strength.

**Tech Stack:** Python 3.12, python-chess 1.11, numpy 2.5, numba 0.67. No new dependencies; the platform installs a fixed stack.

**Spec:** `docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md`

## Global Constraints

- Python 3.12. `uv run ruff check .` and `uv run mypy` (strict, per `pyproject.toml`) must pass clean before every commit.
- ruff line-length 100; rule set `E, F, I, N, UP, B, SIM, RUF`, ignoring `N818`.
- No new third-party dependencies. Only `chess`, `numpy`, `numba` are importable from shipped modules.
- Do not edit `harness/`. Do not edit `nnue_engine.py`, `nnue_features.py`, `nnue_net.py`, `nnue_arch.py`.
- `tools/` never ships. Stockfish stays at `C:/Users/arnav/stockfish/` outside the repo.
- Search scores stay integers in 1/32-centipawn units, positive for the side to move. `MATE = 1_000_000`.
- Every bench run keeps `workers * 2 <= 6` (6 physical cores). Use `--workers 3`.
- The GPU is training in the background. Bench runs share the CPU with its data loader and will slow it; that is accepted, but never launch more than one bench at a time.
- Run tests with `uv run pytest tests/test_search.py -q` and `uv run pytest tests/test_bench.py -q`. The full suite is `uv run pytest -q`.
- Commit messages end with the two trailer lines used on this branch:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01S6mdfxJ2BDHwaBJkZMoY5z
  ```

## The SPRT gate, used by every task in Part 2

After a search change is committed, its strength is judged like this. `HEAD` is the change, `HEAD~1` is the baseline.

```bash
git worktree add ../aichessathon-base HEAD~1
cp weights/nnue.npz ../aichessathon-base/weights/nnue.npz
uv run python tools/elo_bench.py --opponent ../aichessathon-base --sprt --workers 3 --pgn sprt.pgn
git worktree remove ../aichessathon-base
```

The bench prints one line per pair with the LLR and stops at a verdict. Three outcomes:

- **accept**: amend the commit message with the final tally line (`git commit --amend`), keep going.
- **reject**: `git revert HEAD --no-edit`, record the tally in the revert message, move to the next task.
- **undecided at the 400-game ceiling**: treat as reject unless the score is above 52%; if it is, keep the change and note "SPRT undecided, kept on score" in the amended message.

`sprt.pgn` is gitignored output; delete it after reading. Never run two benches at once.

---

## Part 1: measurement

### Task 1: SPRT arithmetic

Pure functions only, so the arithmetic is pinned before it judges anything.

**Files:**
- Modify: `tools/elo_bench.py` (add after `score_margin`, around line 140)
- Create: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Tally` dataclass already in `tools/elo_bench.py` (`wins`, `draws`, `losses`, `games`, `score`).
- Produces: `Sprt(elo0: float = 0.0, elo1: float = 20.0, alpha: float = 0.05, beta: float = 0.05)` frozen dataclass with `lower` and `upper` properties; `expected_score(elo: float) -> float`; `log_likelihood_ratio(tally: Tally, sprt: Sprt) -> float`; `verdict(llr: float, sprt: Sprt) -> str | None` returning `"accept"`, `"reject"` or `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench.py
"""The bench's arithmetic and scheduling, tested without launching a single game."""

import pytest

from tools.elo_bench import Sprt, Tally, expected_score, log_likelihood_ratio, verdict


def test_sprt_bounds_are_symmetric_at_equal_error_rates() -> None:
    sprt = Sprt()
    assert sprt.lower == pytest.approx(-2.944, abs=0.001)
    assert sprt.upper == pytest.approx(2.944, abs=0.001)


def test_expected_score_follows_the_logistic_curve() -> None:
    assert expected_score(0.0) == 0.5
    assert expected_score(400.0) == pytest.approx(10.0 / 11.0)
    assert expected_score(-400.0) == pytest.approx(1.0 / 11.0)


def test_llr_is_zero_before_any_game() -> None:
    assert log_likelihood_ratio(Tally(), Sprt()) == 0.0


def test_llr_grows_with_wins_and_falls_with_losses() -> None:
    sprt = Sprt()
    even = log_likelihood_ratio(Tally(10, 0, 10), sprt)
    better = log_likelihood_ratio(Tally(15, 0, 5), sprt)
    worse = log_likelihood_ratio(Tally(5, 0, 15), sprt)
    assert worse < even < better


def test_a_few_wins_do_not_decide_anything() -> None:
    sprt = Sprt()
    assert verdict(log_likelihood_ratio(Tally(4, 0, 0), sprt), sprt) is None


def test_a_sweep_accepts_and_a_wipeout_rejects() -> None:
    sprt = Sprt()
    assert verdict(log_likelihood_ratio(Tally(40, 0, 0), sprt), sprt) == "accept"
    assert verdict(log_likelihood_ratio(Tally(0, 0, 40), sprt), sprt) == "reject"
    assert verdict(log_likelihood_ratio(Tally(10, 0, 10), sprt), sprt) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bench.py -q`
Expected: FAIL with `ImportError: cannot import name 'Sprt'`.

- [ ] **Step 3: Implement the arithmetic**

Add to `tools/elo_bench.py` directly after `score_margin`:

```python
@dataclass(frozen=True)
class Sprt:
    """Bounds for a sequential test between two hypotheses about the Elo gain.

    H0: the change is worth `elo0`. H1: it is worth `elo1`. The test runs until the
    evidence for one over the other crosses a bound set by the error rates.
    """

    elo0: float = 0.0
    elo1: float = 20.0
    alpha: float = 0.05
    beta: float = 0.05

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)


def expected_score(elo: float) -> float:
    """Score one side is expected to make against an opponent `elo` weaker."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


# Pseudo-count added to each of wins, draws and losses, so a clean sweep has a finite
# variance and the ratio moves smoothly instead of jumping to infinity on game one.
SPRT_REGULARISER = 0.5


def log_likelihood_ratio(tally: Tally, sprt: Sprt) -> float:
    """Generalised SPRT on trinomial results, the form cutechess-cli uses."""
    if tally.games == 0:
        return 0.0
    total = tally.games + 3 * SPRT_REGULARISER
    win = (tally.wins + SPRT_REGULARISER) / total
    draw = (tally.draws + SPRT_REGULARISER) / total
    mean = win + 0.5 * draw
    variance = (win + 0.25 * draw) - mean * mean
    s0 = expected_score(sprt.elo0)
    s1 = expected_score(sprt.elo1)
    return tally.games * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * variance)


def verdict(llr: float, sprt: Sprt) -> str | None:
    """`accept` past the upper bound, `reject` past the lower, None while undecided."""
    if llr >= sprt.upper:
        return "accept"
    if llr <= sprt.lower:
        return "reject"
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -q`
Expected: 6 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run mypy
git add tools/elo_bench.py tests/test_bench.py
git commit -m "feat(tools): add SPRT arithmetic to the bench"
```

---

### Task 2: Node-limited Stockfish

**Files:**
- Modify: `tools/elo_bench.py` — `StockfishAgent.__init__`, `start`, `move` (lines 51-90), `report` (lines 200-220), `main` argument parsing.
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `stockfish_limit(nodes: int | None, time_left_ms: int, increment_ms: int) -> chess.engine.Limit`; `StockfishAgent(path: Path, elo: int | None, nodes: int | None, increment_ms: int)`; `describe_stockfish(elo: int | None, nodes: int | None) -> str`; `report(tally: Tally, label: str, elo: int | None, sprt: Sprt | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
import chess.engine

from tools.elo_bench import describe_stockfish, stockfish_limit


def test_node_limit_ignores_the_clock() -> None:
    limit = stockfish_limit(4_000, time_left_ms=120_000, increment_ms=500)
    assert limit.nodes == 4_000
    assert limit.white_clock is None and limit.black_clock is None


def test_clock_limit_mirrors_our_clock_to_both_sides() -> None:
    limit = stockfish_limit(None, time_left_ms=30_000, increment_ms=500)
    assert limit.nodes is None
    assert limit.white_clock == 30.0 and limit.black_clock == 30.0
    assert limit.white_inc == 0.5 and limit.black_inc == 0.5


def test_opponent_descriptions() -> None:
    assert describe_stockfish(2200, None) == "Stockfish UCI_Elo 2200"
    assert describe_stockfish(None, 4_000) == "Stockfish at 4000 nodes"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bench.py -q`
Expected: FAIL with `ImportError: cannot import name 'describe_stockfish'`.

- [ ] **Step 3: Implement the node limit**

Replace `StockfishAgent` in `tools/elo_bench.py` with:

```python
def stockfish_limit(nodes: int | None, time_left_ms: int, increment_ms: int) -> chess.engine.Limit:
    """Fixed nodes when capped, otherwise our own clock mirrored to both sides.

    A node cap makes Stockfish's strength monotonic and independent of the time
    control, which `UCI_LimitStrength` is not at 120 s + 0.5 s.
    """
    if nodes is not None:
        return chess.engine.Limit(nodes=nodes)
    clock = time_left_ms / 1000.0
    increment = increment_ms / 1000.0
    return chess.engine.Limit(
        white_clock=clock, black_clock=clock, white_inc=increment, black_inc=increment
    )


def describe_stockfish(elo: int | None, nodes: int | None) -> str:
    if nodes is not None:
        return f"Stockfish at {nodes} nodes"
    return f"Stockfish UCI_Elo {elo}"


class StockfishAgent(Agent):
    """Stockfish behind the harness's Agent interface, so the referee stays untouched.

    Either weakened by `UCI_Elo` or at full strength capped to `nodes` per move.
    """

    def __init__(self, path: Path, elo: int | None, nodes: int | None, increment_ms: int) -> None:
        super().__init__([str(path)])
        self.path = path
        self.elo = elo
        self.nodes = nodes
        self.increment_ms = increment_ms
        self._engine: chess.engine.SimpleEngine | None = None

    def start(self, init_budget_s: float) -> None:
        engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        # One core and a small table, matching the constraints our own agent runs under.
        engine.configure({"Threads": 1, "Hash": 16})
        if self.elo is not None:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})
        self._engine = engine

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._engine is None:
            raise RuntimeError("stockfish moved before start")
        board = chess.Board(fen)
        limit = stockfish_limit(self.nodes, time_left_ms, self.increment_ms)
        try:
            played = self._engine.play(board, limit)
        except chess.engine.EngineError as error:
            raise AgentFailure(f"stockfish: {error}") from error
        if played.move is None:
            raise AgentFailure("stockfish returned no move")
        return played.move.uci()

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
```

Replace `report` with:

```python
def report(tally: Tally, label: str, elo: int | None, sprt: Sprt | None = None) -> None:
    margin = score_margin(tally)
    centre = elo_difference(tally.score)
    low = elo_difference(max(0.0, tally.score - margin))
    high = elo_difference(min(1.0, tally.score + margin))

    print(f"\n{tally.games} games vs {label}: +{tally.wins} ={tally.draws} -{tally.losses}")
    print(f"score {tally.score:.1%}")
    if math.isinf(centre):
        bound = "above" if centre > 0 else "below"
        print(f"rating: {bound} {label} (the result does not bracket it)")
    else:
        print(f"rating difference: {centre:+.0f} Elo (1 SE: {low:+.0f} to {high:+.0f})")
        if elo is not None:
            print(f"estimated rating: {elo + centre:.0f} ({elo + low:.0f} to {elo + high:.0f})")
    if sprt is not None:
        llr = log_likelihood_ratio(tally, sprt)
        outcome = verdict(llr, sprt) or "undecided"
        print(
            f"SPRT [{sprt.elo0:g}, {sprt.elo1:g}]: {outcome}, "
            f"LLR {llr:+.2f} in [{sprt.lower:.2f}, {sprt.upper:.2f}]"
        )
    if tally.failures:
        print(f"\nWARNING: {len(tally.failures)} games lost to agent failure: {tally.failures}")
```

In `main`, replace the `--elo` argument with a mutually exclusive group and update the two call sites (`play_pair` construction and `report`). The full `main` is rewritten in Task 3; for this task make the minimal edits:

```python
    parser.add_argument("--stockfish", type=Path, default=None)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--elo", type=int, help="Stockfish UCI_Elo to face")
    opponent.add_argument("--nodes", type=int, help="full-strength Stockfish, N nodes per move")
```

and, after `parse_args`:

```python
    if arguments.stockfish is None:
        parser.error("--stockfish is required with --elo or --nodes")
    label = describe_stockfish(arguments.elo, arguments.nodes)
```

`play_pair` gains a `nodes: int | None` parameter passed through to `StockfishAgent(stockfish, elo, nodes, increment_ms)`; the header print uses `label`; the final call becomes `report(tally, label, arguments.elo)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -q`
Expected: 9 passed.

- [ ] **Step 5: Smoke-test two games at 1k nodes**

Run: `uv run python tools/elo_bench.py --stockfish C:/Users/arnav/stockfish/stockfish.exe --nodes 1000 --games 2 --base-ms 10000 --increment-ms 100`
Expected: two games finish, the report says "vs Stockfish at 1000 nodes" and prints no "estimated rating" line. If the Stockfish binary has a different filename, `ls C:/Users/arnav/stockfish/` and use what is there.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check . && uv run mypy
git add tools/elo_bench.py tests/test_bench.py
git commit -m "feat(tools): bench against node-limited Stockfish"
```

---

### Task 3: Self-play opponent and the SPRT loop

**Files:**
- Modify: `tools/elo_bench.py` — `play_pair`, `main`; add `check_agent_dir`, `run_pairs`.
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Sprt`, `log_likelihood_ratio`, `verdict` (Task 1); `StockfishAgent`, `describe_stockfish`, `report` (Task 2); `harness.sandbox.local`, `Agent`.
- Produces: `check_agent_dir(directory: Path) -> Path`; `run_pairs[T](openings: list[str], workers: int, play: Callable[[str], T], on_pair: Callable[[T], bool]) -> None`; `play_pair(agent_dir: Path, make_opponent: Callable[[], Agent], fen: str, base_ms: int, increment_ms: int) -> list[tuple[Outcome, bool]]`; module constants `SPRT_BASE_MS = 10_000`, `SPRT_INCREMENT_MS = 500`, `SPRT_MAX_GAMES = 400`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`:

```python
from pathlib import Path

from tools.elo_bench import check_agent_dir, run_pairs


def test_run_pairs_stops_scheduling_after_the_verdict() -> None:
    played: list[str] = []
    seen: list[str] = []

    def play(fen: str) -> str:
        played.append(fen)
        return fen

    def on_pair(result: str) -> bool:
        seen.append(result)
        return len(seen) >= 3

    run_pairs([str(i) for i in range(20)], workers=2, play=play, on_pair=on_pair)
    assert len(seen) >= 3
    # Only pairs already in flight when the verdict landed may finish after it.
    assert len(played) <= 3 + 2 - 1
    assert set(seen) == set(played)


def test_run_pairs_plays_everything_when_nothing_stops_it() -> None:
    seen: list[str] = []

    def on_pair(result: str) -> bool:
        seen.append(result)
        return False

    run_pairs(["a", "b", "c"], workers=2, play=lambda fen: fen, on_pair=on_pair)
    assert sorted(seen) == ["a", "b", "c"]


def test_agent_dir_must_hold_agent_and_weights(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="agent.py"):
        check_agent_dir(tmp_path)
    (tmp_path / "agent.py").write_text("")
    with pytest.raises(SystemExit, match="nnue.npz"):
        check_agent_dir(tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "nnue.npz").write_bytes(b"")
    assert check_agent_dir(tmp_path) == tmp_path.resolve()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bench.py -q`
Expected: FAIL with `ImportError: cannot import name 'check_agent_dir'`.

- [ ] **Step 3: Implement the opponent factory, scheduler and SPRT loop**

Add these imports at the top of `tools/elo_bench.py` (keep the existing ones; drop `threading` and the bare `ThreadPoolExecutor` import if they become unused):

```python
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
```

and after the `harness` imports:

```python
from nnue_arch import WEIGHTS_FILE  # noqa: E402
```

Add near the other module constants:

```python
# Self-play SPRT runs on a short base with the platform's real increment: the question
# is which of two versions is stronger, and a short clock answers it in a fraction of
# the time, while the real increment keeps the agent's time manager honest (at 0.1 s it
# overspends by ~200 ms a move and flags in long games, which is pure noise).
SPRT_BASE_MS = 10_000
SPRT_INCREMENT_MS = 500
SPRT_MAX_GAMES = 400
```

Add after `opening_positions`:

```python
def check_agent_dir(directory: Path) -> Path:
    """Fail early, by name, rather than let an opponent crash at init and score a loss."""
    directory = directory.resolve()
    if not (directory / "agent.py").is_file():
        raise SystemExit(f"{directory} has no agent.py")
    weights = directory / "weights" / WEIGHTS_FILE
    if not weights.is_file():
        raise SystemExit(
            f"{directory} has no weights/{WEIGHTS_FILE}; it is gitignored, copy it from this repo"
        )
    return directory


def run_pairs[T](
    openings: list[str],
    workers: int,
    play: Callable[[str], T],
    on_pair: Callable[[T], bool],
) -> None:
    """Play openings on `workers` threads until they run out or `on_pair` returns True.

    `on_pair` runs on the calling thread, so it needs no lock. Pairs already in flight
    when it asks to stop are allowed to finish and are reported too.
    """
    remaining = iter(openings)
    stop = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[T]] = set()
        while True:
            while not stop and len(pending) < workers:
                fen = next(remaining, None)
                if fen is None:
                    stop = True
                    break
                pending.add(pool.submit(play, fen))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                if on_pair(future.result()):
                    stop = True
```

Replace `play_pair` with:

```python
def play_pair(
    agent_dir: Path,
    make_opponent: Callable[[], Agent],
    fen: str,
    base_ms: int,
    increment_ms: int,
) -> list[tuple[Outcome, bool]]:
    """The same opening twice, our agent as white then as black."""
    played: list[tuple[Outcome, bool]] = []
    for we_are_white in (True, False):
        us = local(agent_dir)
        them = make_opponent()
        white, black = (us, them) if we_are_white else (them, us)
        outcome = play_match(white, black, base_ms, increment_ms, PLY_CAP, start_fen=fen)
        played.append((outcome, we_are_white))
    return played
```

Replace `main` entirely:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--elo", type=int, help="Stockfish UCI_Elo to face")
    opponent.add_argument("--nodes", type=int, help="full-strength Stockfish, N nodes per move")
    opponent.add_argument("--opponent", type=Path, help="another agent directory to face")
    parser.add_argument("--stockfish", type=Path, default=None)
    parser.add_argument("--games", type=int, default=None, help="rounded up to a colour pair")
    parser.add_argument("--sprt", action="store_true", help="stop at a verdict, not a count")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=20.0)
    parser.add_argument("--base-ms", type=int, default=None)
    parser.add_argument("--increment-ms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pgn", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="colour pairs to play concurrently; keep workers*2 within the physical "
        "core count or the wall clock the referee measures stops being honest",
    )
    arguments = parser.parse_args()

    sprt = Sprt(arguments.elo0, arguments.elo1) if arguments.sprt else None
    base_ms = arguments.base_ms or (SPRT_BASE_MS if sprt else BASE_MS)
    increment_ms = arguments.increment_ms or (SPRT_INCREMENT_MS if sprt else INCREMENT_MS)
    games = arguments.games or (SPRT_MAX_GAMES if sprt else 20)

    make_opponent: Callable[[], Agent]
    if arguments.opponent is not None:
        opponent_dir = check_agent_dir(arguments.opponent)
        label = f"agent at {opponent_dir}"

        def make_opponent() -> Agent:
            return local(opponent_dir)
    else:
        if arguments.stockfish is None:
            parser.error("--stockfish is required with --elo or --nodes")
        stockfish, elo, nodes = arguments.stockfish, arguments.elo, arguments.nodes
        label = describe_stockfish(elo, nodes)

        def make_opponent() -> Agent:
            return StockfishAgent(stockfish, elo, nodes, increment_ms)

    check_agent_dir(arguments.agent)
    pairs = max(1, (games + 1) // 2)
    openings = opening_positions(pairs, arguments.seed, Balancer())

    print(
        f"agent vs {label} | up to {pairs * 2} games "
        f"| {base_ms / 1000:.0f}s + {increment_ms / 1000:.1f}s "
        f"| {arguments.workers} concurrent"
        + (f" | SPRT [{sprt.elo0:g}, {sprt.elo1:g}]" if sprt else ""),
        flush=True,
    )

    tally = Tally()
    pgn_games: list[str] = []
    done = 0

    def play(fen: str) -> list[tuple[Outcome, bool]]:
        return play_pair(arguments.agent, make_opponent, fen, base_ms, increment_ms)

    def on_pair(played: list[tuple[Outcome, bool]]) -> bool:
        nonlocal tally, done
        for outcome, we_were_white in played:
            tally = record(tally, outcome, we_were_white)
            pgn_games.append(outcome.pgn)
        done += 1
        line = f"pair {done}/{pairs}: +{tally.wins} ={tally.draws} -{tally.losses} ({tally.score:.1%})"
        if sprt is None:
            print(line, flush=True)
            return False
        llr = log_likelihood_ratio(tally, sprt)
        print(f"{line} LLR {llr:+.2f} [{sprt.lower:.2f}, {sprt.upper:.2f}]", flush=True)
        return verdict(llr, sprt) is not None

    run_pairs(openings, arguments.workers, play, on_pair)

    if arguments.pgn is not None:
        arguments.pgn.write_text("\n\n".join(pgn_games), encoding="utf-8")

    report(tally, label, arguments.elo, sprt)
```

Update the module docstring's usage lines to:

```
Usage:
    uv run python tools/elo_bench.py --stockfish PATH --nodes 4000 --games 20
    uv run python tools/elo_bench.py --opponent ../aichessathon-base --sprt --workers 3
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -q`
Expected: 12 passed.

- [ ] **Step 5: Smoke-test self-play for one pair**

```bash
git worktree add ../aichessathon-base HEAD
cp weights/nnue.npz ../aichessathon-base/weights/nnue.npz
uv run python tools/elo_bench.py --opponent ../aichessathon-base --games 2 --sprt
git worktree remove ../aichessathon-base
```

Expected: header says `agent vs agent at ...`, one pair plays at 10 s + 0.5 s, the pair line carries an LLR, the report ends with an `SPRT [0, 20]: undecided` line.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check . && uv run mypy
git add tools/elo_bench.py tests/test_bench.py
git commit -m "feat(tools): self-play opponents and SPRT stopping for the bench"
```

---

### Task 4: Calibrate the bench and record the baseline

No search code changes. This task proves the harness is unbiased and records the numbers every later task is compared against.

**Files:**
- Create: `tools/nps.py`
- Modify: `docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md` (append to "Session context").

**Interfaces:**
- Consumes: `search.Searcher`, `nnue_engine.load_engine`.
- Produces: `tools/nps.py`, which prints nodes and nodes per second on four fixed positions.

- [ ] **Step 1: Write the speed probe**

```python
# tools/nps.py
"""Nodes per second on four fixed positions, for before-and-after speed comparisons.

Not a strength measure: use the SPRT for that. This only says whether a change made
the search faster or slower per node, which the SPRT cannot separate from smarter.
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import search  # noqa: E402
from nnue_engine import load_engine  # noqa: E402

POSITIONS = {
    "opening": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "middlegame": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
    "tactical": "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0",
    "endgame": "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
}
CLOCK_MS = 60_000  # budget_ms(60_000) is about 2.25 s per position


def main() -> None:
    searcher = search.Searcher(load_engine())
    searcher.warm_up()
    total_nodes = 0
    total_seconds = 0.0
    for name, fen in POSITIONS.items():
        started = time.monotonic()
        searcher.pick(fen, CLOCK_MS)
        elapsed = time.monotonic() - started
        total_nodes += searcher.nodes
        total_seconds += elapsed
        print(f"{name:<11} {searcher.nodes:>8} nodes  {searcher.nodes / elapsed / 1000:6.1f} knps")
    print(f"{'overall':<11} {total_nodes:>8} nodes  {total_nodes / total_seconds / 1000:6.1f} knps")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the baseline**

Run: `uv run python tools/nps.py`
Expected: four lines plus an overall line near 20 knps. Paste the output into the spec's "Session context" section under a heading `### Baseline speed, Task 4`.

- [ ] **Step 3: Self-play sanity run, HEAD against itself**

```bash
git worktree add ../aichessathon-base HEAD
cp weights/nnue.npz ../aichessathon-base/weights/nnue.npz
uv run python tools/elo_bench.py --opponent ../aichessathon-base --sprt --games 100 --workers 3
```

Expected: the run reaches the 100-game ceiling with verdict `undecided`, and the score is between 40% and 60%. If a bound is crossed, the harness has a colour or ordering bias: stop, and check that `play_pair` alternates colours and that both sides are launched the same way, before any pruning task runs. Paste the report under `### Self-play sanity, Task 4`.

- [ ] **Step 4: Node-ladder milestone**

Run: `uv run python tools/elo_bench.py --stockfish C:/Users/arnav/stockfish/stockfish.exe --nodes 4000 --games 20 --workers 3`
Expected: 20 games at 120 s + 0.5 s, no failures. Paste the report under `### Node ladder, Task 4`. Then remove the worktree; every later gate recreates it from `HEAD~1`.

```bash
git worktree remove ../aichessathon-base
```

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run mypy
git add tools/nps.py docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md
git commit -m "feat(tools): speed probe and phase 5 baselines"
```

---

## Part 2: the pruning set

Every task here modifies `search.py` and `tests/test_search.py` only. Tests use the existing `make_searcher()` helper in `tests/test_search.py`. A helper for fixed-depth searches is added in Task 6 and reused after.

### Task 5: Transposition table mate scores and replacement

Correctness: no SPRT gate. Tests pin it.

**Files:**
- Modify: `search.py` — `TTEntry`, `Searcher.__init__`, `pick`, `_negamax` (TT probe and `_store` call), `_store`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: existing `MATE`, `MAX_PLY_LIMIT`, `EXACT`, `LOWER`, `UPPER`.
- Produces: `MATE_THRESHOLD: Final = MATE - MAX_PLY_LIMIT`; `to_tt_score(score: int, ply: int) -> int`; `from_tt_score(score: int, ply: int) -> int`; `TTEntry = tuple[int, int, int, chess.Move | None, int]` (depth, score, bound, move, generation); `Searcher.generation: int`; `Searcher._store(key, depth, score, bound, move, ply)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_mate_scores_are_stored_relative_to_the_node() -> None:
    # A mate two plies below a node at ply 3 is "mate in 5" from the root. Stored
    # relative to the node it is "mate in 2", and read back at ply 1 it is "mate in 3".
    searcher = make_searcher()
    searcher._store("key", depth=3, score=search.MATE - 5, bound=search.EXACT, move=None, ply=3)
    stored = searcher.table["key"][1]
    assert stored == search.MATE - 2
    assert search.from_tt_score(stored, ply=1) == search.MATE - 3


def test_losing_mate_scores_convert_the_same_way() -> None:
    assert search.to_tt_score(-search.MATE + 5, ply=3) == -search.MATE + 2
    assert search.from_tt_score(-search.MATE + 2, ply=1) == -search.MATE + 3


def test_ordinary_scores_pass_through_the_table_unchanged() -> None:
    assert search.to_tt_score(123, ply=7) == 123
    assert search.from_tt_score(-123, ply=7) == -123


def test_shallower_entries_do_not_replace_deeper_ones_from_the_same_search() -> None:
    searcher = make_searcher()
    searcher._store("key", depth=5, score=10, bound=search.EXACT, move=None, ply=0)
    searcher._store("key", depth=2, score=20, bound=search.EXACT, move=None, ply=0)
    assert searcher.table["key"][0] == 5
    searcher.generation += 1
    searcher._store("key", depth=2, score=20, bound=search.EXACT, move=None, ply=0)
    assert searcher.table["key"][0] == 2


def test_each_pick_starts_a_new_generation() -> None:
    searcher = make_searcher()
    before = searcher.generation
    searcher.pick(chess.STARTING_FEN, 300)
    assert searcher.generation == before + 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -q -k "mate_scores or pass_through or shallower or generation"`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'from_tt_score'` and a `TypeError` on the unexpected `ply` keyword.

- [ ] **Step 3: Implement**

In `search.py`, after `MAX_PLY_LIMIT`:

```python
# Scores at or beyond this magnitude are mates, and their distance is meaningful.
MATE_THRESHOLD: Final = MATE - MAX_PLY_LIMIT
```

Change the entry type:

```python
TTEntry = tuple[int, int, int, chess.Move | None, int]  # depth, score, bound, move, generation
```

Add module-level functions after the `TTEntry` line:

```python
def to_tt_score(score: int, ply: int) -> int:
    """Convert a root-relative mate score to a node-relative one for storage.

    A mate scores MATE - (plies from the root). Two paths reaching the same position
    at different plies must agree on the entry, so the table holds MATE - (plies from
    this node) instead, and the probe converts back.
    """
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def from_tt_score(score: int, ply: int) -> int:
    """Inverse of to_tt_score, for a probe at `ply`."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score
```

In `Searcher.__init__`, add `self.generation = 0` after `self.nodes = 0`.

In `pick`, add `self.generation += 1` directly after `self.nodes = 0`.

In `_negamax`, change the probe block to convert on read:

```python
        entry = self.table.get(key)
        if entry is not None:
            stored_depth, raw_score, stored_bound, tt_move, _ = entry
            if stored_depth >= depth:
                stored_score = from_tt_score(raw_score, ply)
                if stored_bound == EXACT:
                    return stored_score
                if stored_bound == LOWER and stored_score >= beta:
                    return stored_score
                if stored_bound == UPPER and stored_score <= alpha:
                    return stored_score
```

Change the store call at the end of `_negamax` to `self._store(key, depth, best_score, bound, best_move, ply)`, and replace `_store`:

```python
    def _store(
        self,
        key: Hashable,
        depth: int,
        score: int,
        bound: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        existing = self.table.get(key)
        if existing is not None:
            if existing[4] == self.generation and existing[0] > depth:
                return  # a deeper result from this very search outranks a shallower one
        elif len(self.table) >= TT_MAX_ENTRIES:
            # Crude, but it bounds memory well inside 2 GB and costs nothing on the
            # hot path. Refilling is cheap next to running out of memory.
            self.table.clear()
        self.table[key] = (depth, to_tt_score(score, ply), bound, move, self.generation)
```

Replace the three existing `MATE - MAX_PLY_LIMIT` expressions in `pick` and `_negamax` with `MATE_THRESHOLD`.

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 31 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "fix(search): store mate scores relative to the node and prefer deeper entries"
```

---

### Task 6: Principal variation search

**Files:**
- Modify: `search.py` — `_search_root`, `_negamax` move loop.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: Task 5.
- Produces: `Searcher._search_root(depth: int, previous_best: chess.Move | None, alpha: int = -2 * MATE, beta: int = 2 * MATE) -> tuple[int, chess.Move | None]`; `Searcher._pvs_child(index, depth, ply, alpha, beta, reduction) -> int`; test helper `fixed_depth(searcher, fen, depth, previous_best=None, alpha=-2*MATE, beta=2*MATE)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search.py`:

```python
def fixed_depth(
    searcher: search.Searcher,
    fen: str,
    depth: int,
    previous_best: chess.Move | None = None,
    alpha: int = -2 * search.MATE,
    beta: int = 2 * search.MATE,
) -> tuple[int, chess.Move | None]:
    """Search one depth with no clock pressure, for tests that pin search mechanics."""
    searcher.engine.set_position(fen)
    searcher._deadline = time_module.monotonic() + 60.0
    searcher._path.clear()
    searcher.nodes = 0
    return searcher._search_root(depth, previous_best, alpha, beta)


def test_pvs_recovers_a_better_move_after_a_quiet_first_move() -> None:
    # The first move is searched with the full window and a later one with a null
    # window; that one fails high and must be re-searched to be trusted as best.
    searcher = make_searcher()
    fen = "3q3k/8/8/8/8/8/8/3RK3 w - - 0 1"
    quiet_first = chess.Move.from_uci("e1e2")
    score, move = fixed_depth(searcher, fen, 3, previous_best=quiet_first)
    assert move == chess.Move.from_uci("d1d8")
    assert score > 500 * 32  # a free queen, in 1/32 cp
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_search.py -q -k pvs`
Expected: FAIL with `TypeError: _search_root() takes 3 positional arguments but 5 were given`.

- [ ] **Step 3: Implement PVS at the root and in the tree**

Replace `_search_root`:

```python
    def _search_root(
        self,
        depth: int,
        previous_best: chess.Move | None,
        alpha: int = -2 * MATE,
        beta: int = 2 * MATE,
    ) -> tuple[int, chess.Move | None]:
        board = self.engine.board
        best_score = -2 * MATE
        best_move: chess.Move | None = None
        for index, move in enumerate(self.ordered_moves(board, 0, previous_best)):
            self.engine.push(move)
            try:
                score = self._pvs_child(index, depth - 1, 1, alpha, beta, reduction=0)
            finally:
                self.engine.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_move

    def _pvs_child(
        self, index: int, depth: int, ply: int, alpha: int, beta: int, reduction: int
    ) -> int:
        """Score the child already pushed, with the principal-variation scheme.

        The first move gets the full window. Every later move is first asked only
        "are you better than alpha?" with a null window, which is much cheaper; only a
        yes earns a full search. A reduced move that says yes is first confirmed at full
        depth with the null window, then, if still yes, with the full window.
        """
        if index == 0:
            return -self._negamax(depth, ply, -beta, -alpha)
        score = -self._negamax(depth - reduction, ply, -alpha - 1, -alpha)
        if reduction and score > alpha:
            score = -self._negamax(depth, ply, -alpha - 1, -alpha)
        if alpha < score < beta:
            score = -self._negamax(depth, ply, -beta, -alpha)
        return score
```

In `_negamax`, replace the body of the `try:` inside the move loop (from `reduction = 0` to the end of the LMR re-search) with:

```python
                    reduction = 0
                    if (
                        depth >= LMR_MIN_DEPTH
                        and index >= LMR_MIN_MOVES
                        and not tactical
                        and not in_check
                        and not self.engine.board.is_check()
                    ):
                        reduction = self._late_move_reduction(depth, index)
                    score = self._pvs_child(index, depth - 1, ply + 1, alpha, beta, reduction)
```

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 32 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): principal variation search"
```

- [ ] **Step 6: SPRT gate**

Run the gate from the top of this document. Record the verdict in the commit message (amend) or revert. Also run `uv run python tools/nps.py` and note the overall knps next to the Task 4 baseline in the spec's session context.

---

### Task 7: Aspiration windows

**Files:**
- Modify: `search.py` — constants, `pick`; add `aspiration_window`, `widen`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `_search_root(depth, previous_best, alpha, beta)` and `fixed_depth` from Task 6; `SCORE_PER_CP` from `nnue_arch`.
- Produces: `ASPIRATION_MIN_DEPTH: Final = 5`; `ASPIRATION_WINDOW: Final = 50 * SCORE_PER_CP`; `aspiration_window(score: int, depth: int) -> tuple[int, int, int]` returning `(alpha, beta, window)`; `widen(alpha: int, beta: int, score: int, window: int) -> tuple[int, int, int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_aspiration_window_is_fully_open_at_shallow_depth() -> None:
    alpha, beta, _ = search.aspiration_window(100, search.ASPIRATION_MIN_DEPTH - 1)
    assert (alpha, beta) == (-2 * search.MATE, 2 * search.MATE)


def test_aspiration_window_brackets_the_previous_score() -> None:
    alpha, beta, window = search.aspiration_window(100, search.ASPIRATION_MIN_DEPTH)
    assert alpha < 100 < beta
    assert beta - alpha == 2 * window == 2 * search.ASPIRATION_WINDOW


def test_widen_opens_only_the_failed_side_and_doubles_the_window() -> None:
    alpha, beta, window = search.widen(-1600, 1600, score=-1700, window=1600)
    assert window == 3200
    assert alpha == -1700 - 3200
    assert beta == 1600
    alpha, beta, window = search.widen(alpha, beta, score=1601, window=window)
    assert window == 6400
    assert beta == 1601 + 6400
    assert alpha == -1700 - 3200


def test_widening_reaches_the_full_window_and_stops() -> None:
    alpha, beta, window = -1600, 1600, 1600
    for _ in range(40):
        alpha, beta, window = search.widen(alpha, beta, score=alpha, window=window)
    assert alpha == -2 * search.MATE
    assert beta == 1600


def test_root_search_fails_high_outside_a_narrow_window() -> None:
    searcher = make_searcher()
    narrow = 100 * 32
    fen = "3q3k/8/8/8/8/8/8/3RK3 w - - 0 1"
    score, _ = fixed_depth(searcher, fen, 3, alpha=-narrow, beta=narrow)
    assert score >= narrow
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -q -k "aspiration or widen or narrow_window"`
Expected: the first four FAIL with `AttributeError`; the last passes already (fail-soft root) and stays as a guard.

- [ ] **Step 3: Implement**

Add `from nnue_arch import SCORE_PER_CP` to the imports in `search.py`. Add constants after the LMR block:

```python
# Aspiration windows. From ASPIRATION_MIN_DEPTH on, the root searches a narrow window
# around the previous iteration's score; a fail outside it widens that side and retries.
ASPIRATION_MIN_DEPTH: Final = 5
ASPIRATION_WINDOW: Final = 50 * SCORE_PER_CP
```

Add module-level functions after `from_tt_score`:

```python
def aspiration_window(score: int, depth: int) -> tuple[int, int, int]:
    """(alpha, beta, window) to open iteration `depth` with, given the last score."""
    if depth < ASPIRATION_MIN_DEPTH:
        return -2 * MATE, 2 * MATE, 2 * MATE
    return score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW, ASPIRATION_WINDOW


def widen(alpha: int, beta: int, score: int, window: int) -> tuple[int, int, int]:
    """Open the side the search failed on, doubling the window, up to the full range."""
    window *= 2
    if score <= alpha:
        alpha = max(score - window, -2 * MATE)
    else:
        beta = min(score + window, 2 * MATE)
    return alpha, beta, window
```

Replace the iterative-deepening loop in `pick`:

```python
        best = moves[0]
        score = 0
        for depth in range(1, MAX_DEPTH):
            alpha, beta, window = aspiration_window(score, depth)
            try:
                while True:
                    score, move = self._search_root(depth, best, alpha, beta)
                    if alpha < score < beta or (alpha == -2 * MATE and beta == 2 * MATE):
                        break
                    if move is not None:
                        best = move  # the move that failed high is the best lead we have
                    alpha, beta, window = widen(alpha, beta, score, window)
            except SearchAborted:
                break  # discard this depth entirely; it has a biased best move
            if move is not None:
                best = move
            if abs(score) >= MATE_THRESHOLD:
                break  # a forced mate is as good as it gets
        return best
```

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 37 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): aspiration windows at the root"
```

- [ ] **Step 6: SPRT gate**

Run the gate. Amend or revert.

---

### Task 8: Check extensions

**Files:**
- Modify: `search.py` — `_negamax` move loop; add `child_depth`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `_pvs_child` from Task 6.
- Produces: `child_depth(depth: int, reduction: int, gives_check: bool) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search.py`:

```python
def test_a_checking_move_is_searched_one_ply_deeper() -> None:
    assert search.child_depth(4, reduction=0, gives_check=True) == 4
    assert search.child_depth(4, reduction=0, gives_check=False) == 3
    # A reduced move never gives check (LMR skips them), so the two never combine.
    assert search.child_depth(4, reduction=1, gives_check=False) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_search.py -q -k checking_move`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'child_depth'`.

- [ ] **Step 3: Implement**

Add after `widen` in `search.py`:

```python
def child_depth(depth: int, reduction: int, gives_check: bool) -> int:
    """Remaining depth for a child: one less, less any reduction, plus one for a check.

    Checks are forcing, so a line of them is cheap to follow and expensive to cut
    short: the horizon lands mid-combination and the evaluation sees a lost king
    hunt as a material lead.
    """
    return depth - 1 - reduction + (1 if gives_check else 0)
```

In `_negamax`'s move loop, replace the reduction block and the `_pvs_child` call with:

```python
                    gives_check = self.engine.board.is_check()
                    reduction = 0
                    if (
                        depth >= LMR_MIN_DEPTH
                        and index >= LMR_MIN_MOVES
                        and not tactical
                        and not in_check
                        and not gives_check
                    ):
                        reduction = self._late_move_reduction(depth, index)
                    score = self._pvs_child(
                        index,
                        child_depth(depth, 0, gives_check),
                        ply + 1,
                        alpha,
                        beta,
                        reduction,
                    )
```

(`_pvs_child` subtracts the reduction itself for the null-window probe, so it receives the unreduced child depth.)

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 38 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): extend checking moves by one ply"
```

- [ ] **Step 6: SPRT gate**

Run the gate. Amend or revert.

---

### Task 9: Reverse futility pruning

**Files:**
- Modify: `search.py` — constants, `_negamax`; add `reverse_futility_cutoff`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `MATE_THRESHOLD`, `SCORE_PER_CP`.
- Produces: `RFP_MAX_DEPTH: Final = 3`; `RFP_MARGIN: Final = 120 * SCORE_PER_CP`; `reverse_futility_cutoff(static: int, depth: int, beta: int, in_check: bool) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_reverse_futility_cuts_only_a_comfortable_lead_at_low_depth() -> None:
    margin = search.RFP_MARGIN
    assert search.reverse_futility_cutoff(static=margin + 1, depth=1, beta=0, in_check=False)
    assert not search.reverse_futility_cutoff(static=margin - 1, depth=1, beta=0, in_check=False)
    # The margin grows with depth: what cuts at depth 1 does not at depth 2.
    assert not search.reverse_futility_cutoff(static=margin + 1, depth=2, beta=0, in_check=False)
    assert search.reverse_futility_cutoff(static=2 * margin + 1, depth=2, beta=0, in_check=False)


def test_reverse_futility_never_fires_in_check_deep_or_near_mate() -> None:
    huge = 10_000 * 32
    too_deep = search.RFP_MAX_DEPTH + 1
    assert not search.reverse_futility_cutoff(huge, depth=1, beta=0, in_check=True)
    assert not search.reverse_futility_cutoff(huge, depth=too_deep, beta=0, in_check=False)
    mate = search.MATE_THRESHOLD
    assert not search.reverse_futility_cutoff(huge, depth=1, beta=mate, in_check=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -q -k reverse_futility`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Constants after the aspiration block:

```python
# Reverse futility pruning: near the leaves, a static evaluation comfortably above
# beta is trusted without searching, since a few plies rarely overturn a big lead.
RFP_MAX_DEPTH: Final = 3
RFP_MARGIN: Final = 120 * SCORE_PER_CP
```

Function after `child_depth`:

```python
def reverse_futility_cutoff(static: int, depth: int, beta: int, in_check: bool) -> bool:
    """Whether the static evaluation alone settles this node."""
    return (
        not in_check
        and 0 < depth <= RFP_MAX_DEPTH
        and beta < MATE_THRESHOLD
        and static - RFP_MARGIN * depth >= beta
    )
```

In `_negamax`, move `in_check = board.is_check()` up to directly after the `if depth <= 0 or ply >= MAX_PLY_LIMIT:` quiescence return, and insert after it:

```python
        if depth <= RFP_MAX_DEPTH and not in_check and beta < MATE_THRESHOLD:
            static = self.engine.evaluate()
            if reverse_futility_cutoff(static, depth, beta, in_check):
                return static
```

Remove the later, now duplicate, `in_check = board.is_check()` line.

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 40 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): reverse futility pruning"
```

- [ ] **Step 6: SPRT gate**

Run the gate. Amend or revert.

---

### Task 10: Staged move generation

The one speed item. Accepted on `tools/nps.py` first, then SPRT.

**Files:**
- Modify: `search.py` — `is_draw`, `_move_score`, `ordered_moves`, `_negamax`; add `staged_moves`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `_capture_score`, `killers`, `history` already on `Searcher`.
- Produces: `Searcher.staged_moves(board, ply, tt_move) -> Iterator[tuple[chess.Move, bool]]` yielding `(move, tactical)`; `Searcher.is_draw(board, ply, key: Hashable | None = None) -> bool`; `ordered_moves` keeps its signature and becomes a list over `staged_moves`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_staged_moves_yield_tt_move_then_captures_then_quiets() -> None:
    searcher = make_searcher()
    board = chess.Board("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1")
    tt_move = chess.Move.from_uci("e1e2")
    staged = list(searcher.staged_moves(board, 0, tt_move))
    moves = [move for move, _ in staged]
    assert moves[0] == tt_move
    assert staged[1] == (chess.Move.from_uci("d1d8"), True)
    assert all(not tactical for _, tactical in staged[2:])


def test_staged_moves_skip_an_illegal_tt_move() -> None:
    searcher = make_searcher()
    board = chess.Board("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1")
    bogus = chess.Move.from_uci("a1a8")
    moves = [move for move, _ in searcher.staged_moves(board, 0, bogus)]
    assert bogus not in moves
    assert moves[0] == chess.Move.from_uci("d1d8")


def test_killers_come_after_captures_and_before_other_quiets() -> None:
    searcher = make_searcher()
    board = chess.Board("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1")
    killer = chess.Move.from_uci("e1e2")
    searcher.killers[0] = [killer]
    moves = [move for move, _ in searcher.staged_moves(board, 0, None)]
    assert moves[0] == chess.Move.from_uci("d1d8")
    assert moves[1] == killer


def test_staged_moves_are_a_permutation_of_the_legal_moves() -> None:
    searcher = make_searcher()
    rng = random.Random(3)
    for _ in range(30):
        board = chess.Board()
        for _ in range(rng.randint(0, 30)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        tt_move = rng.choice(list(board.legal_moves)) if board.legal_moves else None
        moves = [move for move, _ in searcher.staged_moves(board, 0, tt_move)]
        assert sorted(m.uci() for m in moves) == sorted(m.uci() for m in board.legal_moves)
        assert len(set(moves)) == len(moves)


def test_is_draw_accepts_a_precomputed_key() -> None:
    searcher = make_searcher()
    board = chess.Board()
    searcher.note_root_position(board)
    assert searcher.is_draw(board, 1, board._transposition_key())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -q -k "staged or precomputed"`
Expected: FAIL with `AttributeError: 'Searcher' object has no attribute 'staged_moves'` and a `TypeError` on `is_draw`.

- [ ] **Step 3: Implement**

Change the import to `from collections.abc import Hashable, Iterator`.

Replace `is_draw`:

```python
    def is_draw(self, board: chess.Board, ply: int, key: Hashable | None = None) -> bool:
        """Whether this node should score as a draw. Never true at the root."""
        if ply == 0:
            return False
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return True
        if key is None:
            key = board._transposition_key()
        return key in self._path or key in self.game_history
```

Delete `_move_score`. Replace `ordered_moves` with:

```python
    def staged_moves(
        self, board: chess.Board, ply: int, tt_move: chess.Move | None
    ) -> Iterator[tuple[chess.Move, bool]]:
        """Legal moves in stages, best guess first, each tagged tactical or not.

        Most nodes cut off on the first move or two, so generating and sorting every
        legal move up front is wasted at most of them. Each stage is generated only
        if the previous stages did not cut off.
        """
        yielded: set[chess.Move] = set()
        if tt_move is not None and board.is_legal(tt_move):
            yielded.add(tt_move)
            yield tt_move, board.is_capture(tt_move) or tt_move.promotion is not None

        own_pawns = board.pawns & board.occupied_co[board.turn]
        tactical = list(board.generate_legal_captures())
        tactical += [
            move
            for move in board.generate_legal_moves(own_pawns, chess.BB_BACKRANKS)
            if move.promotion is not None and not board.is_capture(move)
        ]
        tactical.sort(key=lambda move: self._capture_score(board, move), reverse=True)
        for move in tactical:
            if move not in yielded:
                yielded.add(move)
                yield move, True

        for killer in self.killers[ply]:
            if (
                killer not in yielded
                and killer.promotion is None
                and board.is_legal(killer)
                and not board.is_capture(killer)
            ):
                yielded.add(killer)
                yield killer, False

        quiets = [move for move in board.generate_legal_moves() if move not in yielded]
        quiets.sort(
            key=lambda move: self.history.get((move.from_square, move.to_square), 0),
            reverse=True,
        )
        for move in quiets:
            yield move, False

    def ordered_moves(
        self, board: chess.Board, ply: int, tt_move: chess.Move | None
    ) -> list[chess.Move]:
        """Every legal move, best guess first."""
        return [move for move, _ in self.staged_moves(board, ply, tt_move)]
```

In `_negamax`:

- move `key = board._transposition_key()` above the draw check and call `self.is_draw(board, ply, key)`;
- replace `moves = self.ordered_moves(board, ply, tt_move)` and the `if not moves:` block with:

```python
        if not board.legal_moves:
            return -MATE + ply if in_check else 0
```

- replace `for index, move in enumerate(moves):` and the following `tactical = ...` line with:

```python
            for index, (move, tactical) in enumerate(self.staged_moves(board, ply, tt_move)):
```

`in_check` is already computed above the RFP block (Task 9). Update the null-move comment to: "`legal_moves` is checked first so a stalemate cannot be mistaken for a fail-high; the check stops at the first legal move, so it is cheap."

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 45 passed. The existing `test_tt_move_is_ordered_first`, `test_captures_precede_quiet_moves`, `test_mvv_lva_prefers_the_more_valuable_victim` and `test_ordered_moves_is_a_permutation_of_the_legal_moves` all still pass through `ordered_moves`.

- [ ] **Step 5: Measure speed**

Run: `uv run python tools/nps.py`
Expected: overall knps above the Task 4 baseline. If it is not, look at `board.is_legal` on the TT move and at the `yielded` set; both are new per-node costs. Record the numbers in the spec's session context.

- [ ] **Step 6: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md
git commit -m "perf(search): staged move generation and one key per node"
```

- [ ] **Step 7: SPRT gate**

Run the gate. Amend or revert.

---

### Task 11: Delta pruning in quiescence

**Files:**
- Modify: `search.py` — constants, `_quiescence`; add `delta_pruned`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `SCORE_PER_CP`.
- Produces: `PIECE_CP: Final` mapping piece type to centipawns; `DELTA_MARGIN: Final = 200 * SCORE_PER_CP`; `delta_pruned(static: int, victim: chess.PieceType, alpha: int) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_delta_pruning_skips_captures_that_cannot_reach_alpha() -> None:
    pawn = search.PIECE_CP[chess.PAWN] * 32
    queen = search.PIECE_CP[chess.QUEEN] * 32
    alpha = 1000 * 32
    assert search.delta_pruned(static=0, victim=chess.PAWN, alpha=alpha)
    assert not search.delta_pruned(static=0, victim=chess.QUEEN, alpha=alpha)
    # Exactly on the margin is kept: the margin is the benefit of the doubt.
    on_margin = alpha - pawn - search.DELTA_MARGIN
    assert not search.delta_pruned(static=on_margin, victim=chess.PAWN, alpha=alpha)
    just_under = alpha - queen - search.DELTA_MARGIN - 1
    assert search.delta_pruned(static=just_under, victim=chess.QUEEN, alpha=alpha)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_search.py -q -k delta`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Constants after the RFP block:

```python
# Delta pruning: in quiescence, a capture that could not lift the static evaluation
# to alpha even with this margin on top is not worth searching.
PIECE_CP: Final = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
DELTA_MARGIN: Final = 200 * SCORE_PER_CP
```

Function after `reverse_futility_cutoff`:

```python
def delta_pruned(static: int, victim: chess.PieceType, alpha: int) -> bool:
    """Whether capturing `victim` is hopeless for raising the score to alpha."""
    return static + PIECE_CP[victim] * SCORE_PER_CP + DELTA_MARGIN < alpha
```

In `_quiescence`, declare `static: int | None` before the `if in_check:` branch. In the in-check branch set `static = None`; in the other branch set `static = best_score` right after `best_score = self.engine.evaluate()`. Then in the move loop, before `self.engine.push(move)`:

```python
            if static is not None and move.promotion is None:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if delta_pruned(static, victim, alpha):
                    continue
```

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 46 passed.

- [ ] **Step 5: Lint, type-check, gate, commit**

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): delta pruning in quiescence"
```

- [ ] **Step 6: SPRT gate**

Run the gate. Amend or revert.

---

### Task 12: Static exchange evaluation

**Files:**
- Modify: `search.py` — `staged_moves`, `_quiescence`; add `static_exchange`.
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `PIECE_VALUE` (pawn units), `staged_moves` from Task 10, the delta guard from Task 11.
- Produces: `static_exchange(board: chess.Board, move: chess.Move) -> int` in pawn units for the mover, positive when the exchange wins material.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
def test_see_scores_a_free_pawn_and_a_defended_one() -> None:
    # Rook takes an undefended pawn: +1. Rook takes a pawn defended by a pawn: 1 - 5.
    free = chess.Board("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1")
    assert search.static_exchange(free, chess.Move.from_uci("d1d5")) == 1
    defended = chess.Board("4k3/8/4p3/3p4/8/8/8/3RK3 w - - 0 1")
    assert search.static_exchange(defended, chess.Move.from_uci("d1d5")) == 1 - 5


def test_see_lets_the_defender_decline_a_losing_recapture() -> None:
    # Pawn takes knight; the bishop could recapture but would fall to the rook, so
    # the exchange stops at the knight: +3, not 3 - 1 + 3.
    board = chess.Board("4k3/8/2b5/3n4/2P5/8/8/3RK3 w - - 0 1")
    assert search.static_exchange(board, chess.Move.from_uci("c4d5")) == 3


def test_see_sees_the_rook_behind_the_rook() -> None:
    # Doubled rooks: the front rook takes the knight, the pawn takes it, the back
    # rook takes the pawn through the vacated square: 3 - 5 + 1 = -1.
    board = chess.Board("4k3/8/4p3/3n4/8/8/3R4/3RK3 w - - 0 1")
    assert search.static_exchange(board, chess.Move.from_uci("d2d5")) == 3 - 5 + 1


def test_see_handles_en_passant() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert search.static_exchange(board, chess.Move.from_uci("e5d6")) == 1


def test_losing_captures_are_ordered_after_killers() -> None:
    searcher = make_searcher()
    board = chess.Board("4k3/8/4p3/3p4/8/8/8/3RK3 w - - 0 1")
    killer = chess.Move.from_uci("e1e2")
    searcher.killers[0] = [killer]
    moves = [move for move, _ in searcher.staged_moves(board, 0, None)]
    assert moves.index(killer) < moves.index(chess.Move.from_uci("d1d5"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_search.py -q -k "see_ or losing_captures"`
Expected: FAIL with `AttributeError: module 'search' has no attribute 'static_exchange'`.

- [ ] **Step 3: Implement the swap algorithm**

Add after `delta_pruned`:

```python
def _least_valuable_attacker(
    board: chess.Board, attackers: chess.Bitboard, color: chess.Color
) -> tuple[chess.Bitboard, chess.PieceType] | None:
    order = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    for piece_type in order:
        candidates = attackers & board.pieces_mask(piece_type, color)
        if candidates:
            return candidates & -candidates, piece_type  # lowest set bit
    return None


def static_exchange(board: chess.Board, move: chess.Move) -> int:
    """Material won by `move` after every sensible recapture, in pawn units.

    The classic swap list: each side captures with its least valuable attacker in
    turn, sliders behind the capturer join in as it leaves the line, and each side
    may stop when continuing would lose material. The full list is walked and folded
    from the end; the usual early-exit shortcut is left out because it returns the
    wrong value when a piece behind the capturer joins in.
    """
    square = move.to_square
    victim = board.piece_type_at(square)
    if victim is None and board.is_en_passant(move):
        victim = chess.PAWN
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")

    gain = [PIECE_VALUE[victim] if victim is not None else 0]
    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        captured_square = square + (-8 if board.turn == chess.WHITE else 8)
        occupied &= ~chess.BB_SQUARES[captured_square]
    side = not board.turn
    piece = attacker
    while True:
        # Speculative: what the next capturer nets if it takes and is taken back.
        gain.append(PIECE_VALUE[piece] - gain[-1])
        attackers = board.attackers_mask(side, square, occupied) & occupied
        next_attacker = _least_valuable_attacker(board, attackers, side)
        if next_attacker is None:
            break
        attacker_bb, piece = next_attacker
        occupied &= ~attacker_bb
        side = not side
    gain.pop()  # the last entry assumed a recapture that never came
    while len(gain) > 1:
        last = gain.pop()
        gain[-1] = -max(-gain[-1], last)  # each side may stop instead of continuing
    return gain[0]
```

In `staged_moves`, split the tactical stage into winning and losing captures. Replace the loop after `tactical.sort(...)` with:

```python
        losing: list[chess.Move] = []
        for move in tactical:
            if move in yielded:
                continue
            if move.promotion is None and static_exchange(board, move) < 0:
                losing.append(move)
                continue
            yielded.add(move)
            yield move, True
```

and, after the killers stage and before the quiets, yield the losing captures:

```python
        for move in losing:
            yielded.add(move)
            yield move, True
```

In `_quiescence`, extend the delta-pruning guard so losing captures are skipped when not in check:

```python
            if static is not None and move.promotion is None:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if delta_pruned(static, victim, alpha) or static_exchange(board, move) < 0:
                    continue
```

- [ ] **Step 4: Run the full search tests**

Run: `uv run pytest tests/test_search.py -q`
Expected: 51 passed.

- [ ] **Step 5: Measure speed, then lint, type-check, gate, commit**

Run: `uv run python tools/nps.py`. SEE costs python-chess calls per capture, so a modest knps drop is expected; note the number in the commit message.

```bash
uv run ruff check . && uv run mypy && make gate
git add search.py tests/test_search.py
git commit -m "feat(search): static exchange evaluation for capture ordering and quiescence"
```

- [ ] **Step 6: SPRT gate**

Run the gate. Amend or revert.

---

### Task 13: Close out the phase

**Files:**
- Modify: `docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md` (session context).
- Rebuild: `submission.zip` via `make zip`.

- [ ] **Step 1: Re-run the node ladder**

Run: `uv run python tools/elo_bench.py --stockfish C:/Users/arnav/stockfish/stockfish.exe --nodes 4000 --games 20 --workers 3`
Paste the report under `### Node ladder, after Part 2` in the spec's session context, next to the Task 4 number.

- [ ] **Step 2: Table of verdicts**

Add a table to the session context with one row per Task 6-12: the technique, the SPRT verdict, the final tally, and the overall knps from `tools/nps.py`.

- [ ] **Step 3: Rebuild and gate the submission**

```bash
make gate && make zip
```

Expected: gate clean, `submission.zip` rebuilt with `agent.py` at the root.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-09-04-measurement-and-pruning-design.md
git commit -m "docs: record phase 5 results"
```
