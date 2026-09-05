# Phase 6: a numba bitboard engine

**Written** 2026-09-05, branch `bitboard` off `nnue` at 74a7d63. Follows the phase 5
spec, whose close-out named this rewrite as the one speed lever left.

## Problem

The search runs at 13-22 k nodes per second and reaches depth 5-7 at tournament budget.
Every profile since phase 4 says the same thing: python-chess move generation and
`Board.push`, plus the Python glue around the accumulator update, take most of the time,
and the NNUE evaluation itself takes about 4%. Against full-strength Stockfish capped at
4,000 nodes per move the agent scored 1 draw in 20 games. The phase 5 pruning set bought
every ply that was available without more nodes. What is left is nodes.

Staged move generation was measured neutral in phase 5, which settled where the time goes:
not in ordering or sorting, but in the board itself. Speeding the board up means leaving
python-chess for everything inside the search.

## Goal

A move generator, make-move, evaluation and search loop that all run inside numba
`nopython` code, sharing one set of preallocated numpy arrays, with python-chess used
only at the root to parse the FEN and to check the returned move. Target: at least
150 k nodes per second on the phase 4 profile positions, roughly ten times today, and
otherwise the same search as commit 74a7d63 so the first gate measures speed alone.

## Constraints

- Platform: one core, 2 GB, 120 s + 0.5 s wall clock, torch 2.13 CPU, numpy 2.5,
  python-chess 1.11, numba 0.67, onnxruntime 1.29 and nothing else. Read-only
  filesystem apart from 256 MB at `/tmp`. Native binaries in the zip are rejected, so
  numba's on-disk cache cannot ship; every kernel compiles at import, inside the 60 s
  init budget. All kernels use `cache=False`, as today.
- `harness/` is not edited. `tools/` never ships.
- Style: Python 3.12, ruff and mypy strict clean. The kernel code is the thing a judge
  reads if a game is flagged, so it stays plain: no jitclass, no magic-number tables,
  no generated code.
- Scores stay integers in 1/32 centipawn units, positive for the side to move,
  `MATE = 1_000_000`, exactly as `search.py` has them, so every margin sized in phase 5
  carries over unchanged.
- The search algorithm is ported, not redesigned. Iterative deepening, aspiration
  windows, PVS, TT with node-relative mate scores, null move, LMR, check extension,
  reverse futility, quiescence with delta pruning and SEE, MVV-LVA, killers, history,
  repetition and 50-move draws: the same rules with the same constants. Tuning for the
  new depths is the next phase, gated separately.

## Spikes that settled the design (2026-09-05)

- A self-recursive `@njit` search compiles and runs; numba 0.67 resolves the recursion
  from the non-recursive base case. Mutual recursion is avoided anyway: quiescence is
  the `depth <= 0` branch of the same kernel.
- Reading the clock from inside a kernel through `numba.objmode` costs 0.8 µs. Checked
  every 4,096 nodes it is invisible, and it keeps the time control in the kernel rather
  than forcing the search to return to Python to look at the clock.
- The three existing NNUE kernels compile in 2.8 s. The new kernels are larger; compile
  time is measured in every task, and the design keeps the number of distinct compiled
  functions small (one function per concern, a handful of signatures).

## Architecture

Three new modules, all root-level so they ship, plus a rewritten `search.py`.

```
agent.py         unchanged interface: get_move(fen, time_left_ms) -> uci
search.py        Searcher: time budget, iterative deepening, aspiration loop, root
                 safety check. Owns every array the kernels use. Python only.
search_kernel.py the recursive alpha-beta kernel and its helpers (ordering, SEE,
                 TT probe/store, draw detection). numba only.
bitboard.py      board representation, attack tables, move encoding, pseudo-legal
                 move generation, copy-make, zobrist hashing, FEN in and out.
nnue_bitboard.py HalfKAv2_hm feature indices over the bitboard arrays, accumulator
                 refresh and update, forward pass (moved from nnue_engine.py).
```

`nnue_arch.py` and `nnue_net.py` stay as they are. `nnue_features.py` stays as the
python-chess reference the feature tests compare against. `nnue_engine.py` and the
python-chess search are deleted once the gate passes; until then they are the parity
oracle.

### Board representation (`bitboard.py`)

Copy-make: every ply has its own full board, so unmake is `ply -= 1`, exactly as the
accumulators already work. Per ply the board is:

| array | shape | contents |
| --- | --- | --- |
| `pieces` | `[MAX_PLY, 12]` uint64 | one bitboard per piece code: white P N B R Q K = 0..5, black = 6..11 |
| `occupied` | `[MAX_PLY, 3]` uint64 | white, black, both |
| `mailbox` | `[MAX_PLY, 64]` int8 | piece code on each square, 12 for empty |
| `state` | `[MAX_PLY, 6]` int32 | side to move, castling rights (4 bits, WK WQ BK BQ), en passant square or -1, halfmove clock, king square white, king square black |
| `keys` | `[MAX_PLY]` uint64 | zobrist key of the position at that ply |

Square numbering is python-chess's (a1 = 0, h8 = 63) so FEN conversion and every test
that compares with python-chess need no translation. `MAX_PLY` stays 256.

Attacks: knight, king and pawn attack tables are 64-entry arrays built at import with
plain loops. Sliders use hyperbola quintessence (`o ^ (o - 2r)` along a ray mask and
its reverse) for files, diagonals and anti-diagonals, and an 8 x 64 first-rank lookup
for ranks. No magic bitboards: they need either 128 opaque constants or an import-time
random search, and the speed difference is irrelevant next to everything else.

Zobrist keys: a `[12, 64]` piece table, 16 castling entries, 8 en passant files and a
side-to-move key, drawn from `numpy.random.default_rng(20260905)` at import. The key
is updated incrementally in make and checked against a from-scratch recomputation in
tests.

### Moves

A move is one int32: `from | to << 6 | promotion << 12 | flags << 15`, promotion 0 or
piece type 2..5, flags for capture, en passant, castling, double pawn push. Promotions
generate four moves each. Castling is encoded as the king's two-square move.

Move generation is pseudo-legal with legality checked when the move is made: after the
copy-make, if the mover's king is attacked the move is discarded and the ply is not
entered. Alpha-beta makes only a fraction of the moves it generates, so verifying only
the moves actually made is cheaper than generating strictly legal lists, and it keeps
the generator simple. The only legality done up front is the standard castling rule:
the king may not castle out of, through or into check, and the squares between must be
empty. Move lists live in `moves[MAX_PLY, 256]` int32 with a parallel `scores` array.

The mate and stalemate test is "no legal move was found after trying every generated
move", which the kernel tracks with a counter, the same way the existing search uses an
empty legal-move list.

### Evaluation on bitboards (`nnue_bitboard.py`)

The feature index formula from `nnue_features.py` is a few integer operations on
(perspective, king square, square, piece code), so the accumulator refresh walks the
mailbox and the update computes the handful of added and removed rows straight from the
move: mover from and to, captured piece, castling rook, en passant pawn, promotion
piece. A perspective whose own king moved is refreshed, as today. `_forward` moves here
unchanged; the accumulator and PSQT stacks keep their current shapes and dtypes so
`tests/reference.py` still pins the arithmetic. The result must match
`nnue_engine.Engine.evaluate` exactly on every position of a random game, which is the
test that accepts this module.

### The search kernel (`search_kernel.py`)

One recursive function, `search(depth, ply, alpha, beta, ...)`, taking the board
stacks, the accumulator stacks, the network arrays, the TT arrays, the killers and
history tables, the move lists, and a small `ctrl` int64 array (node count, abort flag,
node limit) plus a float deadline. `depth <= 0` is quiescence. It returns the score;
the best move at the root is written into `ctrl`.

Everything from the current `search.py` maps onto it one to one:

- Draw detection: the ply key stack for repetitions inside the search, a game-history
  key array for positions seen at earlier roots, halfmove clock 100, and insufficient
  material (no pawns, no queens or rooks, at most one minor piece in total).
- TT: `tt_keys` uint64 and `tt_data` int64 arrays of 2^20 entries, the data word
  packing depth, bound, generation, score and move. Index is `key & (size - 1)`. Store
  keeps a deeper entry from the same generation, otherwise replaces, which is the
  current rule without the `clear()`.
- Ordering: TT move, winning captures by MVV-LVA with SEE deciding winning or losing,
  killers, history, losing captures, computed into `scores` and selected by partial
  selection sort, so a node that cuts off on the first move never sorts the rest.
- Null move, LMR, check extension, reverse futility, aspiration and PVS: same
  constants, same conditions. SEE is the phase 5 swap list, on bitboard attackers.
- Time: `ctrl[0]` counts nodes; every 4,096 nodes the kernel reads the clock through
  `objmode` and sets `ctrl[1]` on the deadline. Each recursion level checks the flag
  after a child returns and unwinds. No exceptions cross the kernel boundary.

Root handling stays in Python. `Searcher.pick` parses the FEN with python-chess, fills
ply 0, runs the iterative-deepening and aspiration loop by calling the kernel once per
window, and verifies the move it is about to return against `board.legal_moves`. If the
kernel ever returned an illegal move that is a bug to fix, but the game is not lost to
it: the fallback is the first legal move, logged.

### Memory

Board stacks under 1 MB. Accumulators as today. TT 2^20 entries x 16 bytes = 16 MB.
Weights 7 MB. numba and numpy at import roughly 150 MB. Well inside 2 GB.

## Testing

- **Attack tables:** for every square, and for 2,000 random occupancies, each attack
  function equals python-chess's `attacks_mask` for the same piece and occupancy.
- **Move generation:** perft at depths 1-4 on the six standard positions from the Chess
  Programming Wiki (start, Kiwipete, positions 3-6) equals the published node counts;
  and differential testing on random games: for every position of 200 seeded random
  games, the set of legal moves as UCI strings equals `chess.Board.legal_moves`. This
  is the test that accepts the generator.
- **Make and keys:** after every move of those games, the incremental zobrist key
  equals the from-scratch key, and the board equals `chess.Board` square by square.
- **Evaluation:** exact equality with `nnue_engine.Engine.evaluate` along random games,
  both after incremental updates and after a refresh.
- **Search:** the behavioural tests from `tests/test_search.py` carry over: mate in one,
  mate in two through a quiet move, takes the free queen, legal move in check, never an
  illegal move over 1,000 random positions, respects a tight budget, draws by
  repetition and by the 50-move rule, mate scores relative to the node.
- **Import budget:** a test imports `agent` in a fresh interpreter and asserts it takes
  under 30 s on this machine, leaving a factor of two for the platform's core.
- **Speed:** `tools/nps.py` adapted to the new searcher, run on the four profile
  positions, recorded in this spec.

## Gate

`make gate`, then self-play SPRT against a worktree of 74a7d63 with
`tools/elo_bench.py --opponent ... --sprt --workers 3`, then the node ladder at 4,000
and 16,000 nodes for the record. A ten-times faster search of the same algorithm should
cross the accept bound within a few dozen games. If it does not, the port is wrong
somewhere, and the differential tests are the first place to look.

## Not in this phase

- Retuning margins, late move pruning, futility at frontier nodes, countermove
  history: phase 7, once the new depths are known.
- Pondering: the kernel is written so all state is in arrays it is handed, which is what
  a `nogil` background search needs later, but no thread is started here.
- Magic bitboards, strictly legal generation, Chess960 castling.
- Any change to the net or the training pipeline; net2 trains on the GPU meanwhile.

## Session context

### Rules update noted mid-build (2026-09-05)

The organisers removed pondering: the process is now suspended while the opponent
moves, so each side has the core to itself. The rules page read "120s plus 0.5s per
move, per side, with a 90s init budget before the clock" when fetched during this
session; the user reported the base time as 90 s. Nothing here depends on either
number: the budget derives from the clock the platform hands us, and only 0.6 of the
published increment is claimed. Pondering leaves the roadmap for good.

### Tasks 1-4: what landed

- Perft matches the six published positions to depth 3-4, and the generator agrees
  with python-chess on every position of 200 random games (5,000+ positions).
- Evaluation is bit-exact with `nnue_engine.Engine` on refresh, along 20 random games
  of incremental updates, and across null moves.
- All behavioural search tests pass on the kernel, including a deterministic
  node-limited search, K+Q conversion without a threefold, and 40 random games with no
  illegal move.

### Task 5 measurements (this desktop, GPU trainer sharing the CPU)

Speed on the phase 4 profile positions, `tools/nps.py`, budget ~2.25 s per position:

| position | nodes | knps |
| --- | ---: | ---: |
| opening | 417,792 | 183.6 |
| middlegame | 413,696 | 183.2 |
| tactical | 425,984 | 188.2 |
| endgame | 688,128 | 303.4 |
| overall | 1,945,600 | 214.6 |

Phase 5 measured 13.2 knps overall under the same load (22 knps unloaded): a factor of
ten to fourteen, and at the same 4 s budget the search now sees about 900k nodes where
it saw 90k.

Import time: the first build took 44 s because numba compiled the search kernel three
times, once per distinct literal argument at its call sites (`True`, `False` for the
null-move flag; `-1` for the hash move). Deriving the flags from values and moving the
null-move permission into a per-ply array brought a fresh `import agent` to 24.7 s
(0.6 s Python, ~5 s bitboard kernels, ~3.4 s NNUE kernels, ~15.6 s the search kernel).
`tests/test_import_time.py` bounds it at 45 s locally, half the platform's 90 s.
Numba's optimisation level makes no difference (40-44 s at every level before the
fix), so the cost is type inference and lowering, not LLVM.

### Task 6 gate

`make gate` steps: ruff and mypy clean, two harness games against `baselines/random`
won by checkmate. Self-play SPRT against `../aichessathon-net108`, a worktree at
251b8c5 (the shipped code and net; 74a7d63 differs from it only in docs), at
10 s + 0.5 s with 3 workers, sharing the machine with the net2 trainer:

```
20 games vs agent at ../aichessathon-net108: +18 =2 -0
score 95.0%
rating difference: +512 Elo (1 SE: +414 to +720)
SPRT [0, 20]: accept, LLR +5.09 in [-2.94, 2.94]
```

Twenty games, no losses, no failures. At this clock the old engine reaches depth
4-5 and the new one depth 8-9 from the same code and net; the score is what a
ten-times faster search of the same algorithm should produce.

The zip built by `harness.package` holds `agent.py bitboard.py nnue_arch.py
nnue_bitboard.py nnue_features.py nnue_net.py search.py search_kernel.py
weights/nnue.npz`, 7.21 MB unzipped. From an extracted copy in a fresh interpreter:
import 26.1 s, a middlegame move at a 90 s clock in 3.3 s, an endgame move at a 2 s
clock in 0.3 s.

### Task 7

`nnue_engine.py` is gone. `nnue_bitboard.evaluate_board(net, board)` is the one-off
scoring helper that `tests/test_parity.py`, `tests/test_export.py`,
`tools/verify_export.py` and the bench's opening balancer now use; the parity tests
pin the kernels against `tests/reference.forward_int` directly.
