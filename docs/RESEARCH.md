# Research foundation and implementation map

This document turns the literature we reviewed into hypotheses for an original, competition-legal
engine. It is a guide, not permission to copy Stockfish code or weights. General algorithms and
published theories should be implemented independently and tested in this repository's harness.

## Recommended architecture

The highest-probability design under a one-core Python runtime is a selective alpha-beta search
with a very fast evaluation function:

```text
iterative deepening
  -> aspiration window / principal-variation search
  -> negamax alpha-beta
  -> transposition-table lookup
  -> ordered legal moves
  -> selective pruning, reductions, and extensions
  -> quiescence search
  -> handcrafted or compact learned evaluation
```

Search supplies tactical calculation. Evaluation supplies strategic judgment at stable leaves.
An AlphaZero-style policy/value network with MCTS is valuable background, but its training and
inference costs make it a lower-probability choice here.

## Search theories to implement

### Alpha-beta and move ordering

Knuth and Moore show that alpha-beta returns the minimax result while avoiding work that cannot
affect it. Its best-case tree is dramatically smaller when strong moves are searched first. This
makes move ordering a core algorithmic dependency rather than cosmetic sorting.

Practical order:

1. Previous-iteration principal-variation or transposition-table move.
2. Promotions and promising captures, initially ordered by victim value minus attacker cost.
3. Killer quiet moves that caused cutoffs at the same ply.
4. Quiet moves ranked by a history score updated after cutoffs.
5. Remaining moves.

Use iterative deepening so every completed depth supplies the next depth's first move and always
leaves a legal fallback when time expires.

### Zobrist hashing and transpositions

Zobrist assigns random bit strings to piece-square and state features, then XORs them into a
position key. A move updates the key by XORing out removed features and XORing in added ones.

A transposition entry should record at least the key, searched depth, score, bound type (exact,
lower, or upper), and best move. Mate scores need ply normalization when stored and retrieved.
The position identity must include side to move, castling rights, and relevant en-passant state.

### Quiescence and the horizon

Static evaluation is unreliable halfway through a forcing exchange. At nominal depth zero,
quiescence should continue through tactically forcing moves until it reaches a sufficiently quiet
position. Start with legal captures and promotions, a stand-pat score, alpha-beta bounds, and
correct behavior while in check. Add delta pruning only after correctness tests.

### Selective search

Add these individually after the full-width engine is stable:

- Principal-variation search: use a full window for the expected best move and null windows for
  later moves, re-searching when they improve alpha.
- Null-move pruning: if giving the opponent an artificial extra move still reaches beta at reduced
  depth, prune, with safeguards for check and zugzwang-prone endings.
- Late-move reductions: search low-ranked quiet moves at reduced depth, then re-search surprising
  improvements at full depth.
- Futility or reverse-futility pruning: omit quiet work when a shallow static bound makes changing
  the result implausible.
- Extensions: selectively deepen forcing cases such as check evasions or a singularly strong move.

Every technique trades missed tactics for speed. It must be measured in games and disabled in
positions where its assumptions are unsafe.

## Evaluation theories to implement

Begin with a tapered handcrafted evaluator so bugs and performance are visible:

- Material with separate middlegame and endgame values.
- Piece-square placement and game-phase interpolation.
- Mobility, bishop pair, rook files, passed pawns, doubled/isolated pawns, king shelter, and threats.
- Scores returned from the side-to-move perspective and terminal mate scores kept outside the
  ordinary evaluation range.

A learned evaluator should replace or blend with this baseline only after it wins controlled
matches. The NNUE principle is especially relevant: sparse chess inputs and small changes between
successive positions allow incremental feature accumulation. Shallow integer networks can be much
faster than generic floating-point models.

For a first learned model, prefer a compact position-value network over a full policy/value system.
Train offline, benchmark pure inference on one core, and include loading time in the 60-second init
budget. ONNX is operationally convenient; a custom NumPy/Numba integer evaluator may be faster if
incremental updates are implemented correctly.

## Data and learning lessons

- Existing engines may label offline training positions under the competition rules.
- Deep-search values can supervise a cheaper evaluator: search acts as a teacher whose work is
  amortized into inference.
- Mix searched evaluation targets with game outcomes rather than blindly copying shallow scores.
- Prefer diverse, quiet positions where the label is stable. Positions with unresolved captures,
  forks, or forced checks inject tactical noise that belongs in search.
- Split data by games or position families to avoid near-duplicate leakage.
- Measure calibration and rank accuracy, but select the final model by engine match results.
- TD-Leaf suggests training from principal leaves reached by search rather than only from played
  board states; the evaluator should learn the states it will actually see at search boundaries.

AlphaZero demonstrates that policy/value learning plus tree search can learn chess from self-play.
DeepChess and transformer distillation demonstrate that networks can internalize substantial chess
knowledge. They do not remove the engineering constraint here: a model called thousands of times
per move must be extremely small and fast.

## Experimental method

Stockfish's Fishtest methodology provides the model even though we cannot reproduce its scale:

1. Change one concept at a time.
2. Play paired games from the same opening position with colors reversed.
3. Use a fixed previous version as the opponent.
4. Record wins, draws, losses, time forfeits, crashes, nodes, completed depth, and move latency.
5. Run enough games to distinguish improvement from noise.
6. Confirm promising short-time-control results at the real 120+0.5 control.
7. Keep a change only when evidence supports it or when it clearly improves correctness or
   simplicity without reducing strength.

Paired outcomes are correlated, so paired/pentanomial statistics are preferable to treating every
game as independent. Sequential tests such as SPRT can stop losing or clearly winning experiments
early. SPSA is useful later for simultaneously tuning numeric parameters, but it cannot rescue a
weak architecture or noisy match setup.

## Build order

1. Legal fallback, terminal detection, mate-distance scores, and monotonic clock handling.
2. Iterative-deepening negamax with alpha-beta and a basic tapered evaluation.
3. Quiescence search and tactical regression positions.
4. Zobrist keys and a bounded transposition table.
5. PV/TT, capture, killer, and history move ordering.
6. Principal-variation search and aspiration windows.
7. Null move, late-move reductions, and shallow pruning, one experiment at a time.
8. Evaluation feature improvements supported by ablations.
9. Compact learned evaluation only after the classical engine supplies a robust baseline.
10. Pondering, cross-move table reuse, opening/endgame data, and final packaging hardening.

## Primary and technical sources

- Donald Knuth and Ronald Moore, [An Analysis of Alpha-Beta Pruning](https://doi.org/10.1016/0004-3702(75)90019-3), 1975.
- Albert Zobrist, [A New Hashing Method with Application for Game Playing](https://research.cs.wisc.edu/techreports/1970/TR88.pdf), 1970.
- Jonathan Schaeffer, [The History Heuristic and Alpha-Beta Search Enhancements in Practice](https://doi.org/10.1109/34.42858), 1989.
- Jonathan Baxter, Andrew Tridgell, and Lex Weaver, [Learning to Play Chess Using Temporal Differences](https://doi.org/10.1023/A:1007634325138), 2000.
- Joel Veness et al., [Bootstrapping from Game Tree Search](https://proceedings.neurips.cc/paper/2009/hash/389bc7bb1e1c2a5e7e147703232a88f6-Abstract.html), 2009.
- David Silver et al., [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), 2017.
- Eli David, Nathan Netanyahu, and Lior Wolf, [DeepChess](https://arxiv.org/abs/1711.09667), 2017.
- Dominik Klein, [Neural Networks for Chess](https://arxiv.org/abs/2209.01506), 2022.
- Daniel Tan and Neftali Watkinson Medina, [Study of the Proper NNUE Dataset](https://arxiv.org/abs/2412.17948), 2024.
- Anian Ruoss et al., [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://proceedings.neurips.cc/paper_files/paper/2024/hash/78f0db30c39c850de728c769f42fc903-Abstract-Conference.html), 2024.
- [Official Stockfish documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Home.html), including the search terminology, NNUE, Fishtest, and useful-data sections.

