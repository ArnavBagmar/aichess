# Rounds 13–15 and residual HalfKP gates

## Real ladder results (4 September 2026)

The dashboard logs contain only our moves and clocks, not the opponent's moves. They therefore
cannot reconstruct complete games or support move-by-move engine analysis. They are still useful
for outcome, reliability, repetition, and time-management analysis.

- Round 13, Black vs Magnus CarlSON, Pirc: win by checkmate in 32 agent moves. The engine used
  116.4 seconds and retained 19.6 seconds. The final passed pawn promoted with four instant forced
  moves.
- Round 14, White vs Pwn, Closed Sicilian: draw by threefold in 28 agent moves. The engine used
  115.7 seconds and retained 18.3 seconds. The last four moves repeated `Kh2`, `Kg1`, `Kh2`,
  `Kg1`.
- Round 15, White vs What the fork, Sicilian Dragon: draw by threefold in 39 agent moves. The
  engine used 133.7 seconds (including increments) and retained 5.8 seconds. The final sequence
  alternated `Rf6+` and `Rg6+`.

Aggregate: +1 =2 -0, no crash, flag, illegal move, or initialization failure. The data suggests
good tactical reliability, but also a need for better repetition/conversion judgment. The 5.8
second finish is safe but close enough that time use should continue to be monitored.

### Cross-call repetition correction

The API supplies a FEN on every turn, so a newly constructed `chess.Board` has no game move stack.
The old engine could recognize a repetition inside one search line but could not recognize that an
actual root position had returned across API calls. The engine now remembers its prior choice for
each root. If the identical root returns while the static evaluation is at least +150 cp, it applies
a strong penalty to repeating the same continuation. At equal or negative evaluation it keeps the
normal move, preserving valuable perpetual-check draws. A targeted regression confirms that a
clearly winning repeated root changes continuation. General-play A/B was neutral at +6 =8 -6 over
20 paired games, with no failures; this is retained as a narrowly activated correctness safeguard,
not claimed as measured Elo gain.

## Quantized residual HalfKP

The team-trained 32-unit HalfKP residual network was quantized to int16 and evaluated through a
Numba kernel. The shipped model is `nnue.npz` (about 0.77 MB). It is a learned correction to the
handcrafted evaluation, never a bundled third-party engine.

- Held-out 3,000-position MAE: about 141 cp with the full residual versus 169 cp handcrafted.
- Raw evaluation throughput: about 35k/s with residual versus 44k/s handcrafted.
- Full residual A/B: +4 =8 -8 over 20 fast games (40%); rejected.
- Half residual initial A/B: +9 =3 -8 over 20 fast games (52.5%).
- Half residual confirmation: +17 =14 -9 over 40 fast games (60.0%, pair-aware interval
  51–69%).
- Half residual combined fast A/B: +26 =17 -17 over 60 games (57.5%, approximately +52 Elo).
- Half residual official-clock A/B: +1 =2 -1 over four games (50%), with both pairs neutral.

The 50% blend passes the promotion gate; the full correction does not. This is a concrete example
of why label MAE is subordinate to paired game results.

## Forty-game official 2600 reference sample

Five initial pairs and fifteen precommitted extension pairs were played at the official 120+0.5
clock on forty distinct color assignments. The combined result was +7 =13 -20 (33.75%), or -117
unanchored Elo relative to the local Stockfish 2600 configuration. The pair-aware 95% score
interval was 22.9–44.6%, corresponding to -211 to -38 Elo. If the configured opponent were
perfectly calibrated, that maps to a 2483 point estimate and approximately 2389–2562 interval.
The local opponent uses Stockfish's UCI strength limiter and 100 ms per move, however, so this is
an internal reference—not a certified public rating. There were no agent failures.

Deeper fixed-budget analysis of 1,528 agent moves found competitive-position mean loss of 25.9 cp
in the opening, 43.9 cp in the middlegame, and 30.5 cp in the endgame. This points to middlegame
tactical horizon errors rather than opening selection as the largest immediate weakness.

Two forcing-search experiments were rejected after paired testing:

- PV-only quiet checks: pilot 55%, fresh confirmation 40%, combined 45% over 60 games.
- PV mate-in-one leaf detection: pilot 52.5%, fresh confirmation 48.75%, combined exactly 50% over
  60 games.

Neither is shipped. This guards against pilot-selection bias and preserves the strongest measured
version rather than accumulating plausible but unproven heuristics.

## Deeper-label and incremental-NNUE experiments

A complete 32,607-position realistic Lichess corpus was relabelled by Stockfish 18 at depth 10.
Three independently seeded HalfKP32 residuals reached 144.9--145.5 cp held-out MAE, compared with
163.3 cp for the handcrafted evaluator on the same split. Better static-fit metrics did not
translate reliably into games:

- Seed 05 scored +4 =6 -10 (35%) in its 20-game pilot and was rejected.
- Seed 06 scored +3 =7 -0 (65%) in a 10-game seed screen.
- Seed 07 scored +5 =5 -0 (75%) on the same screen, then +11 =16 -13 (47.5%) in a
  precommitted fresh 40-game confirmation and was rejected.
- A deployment-calibrated model trained specifically for the engine's 50% residual blend reached
  144.1 cp held-out deployed MAE, but scored +3 =9 -8 (37.5%) in its fresh 20-game pilot and was
  rejected.

The training tool now supports `--deployment-blend` and reports validation error for the score
actually used by the engine. This makes future blend experiments correctly calibrated even though
this particular candidate did not pass the game gate.

An incremental HalfKP accumulator was also implemented and correctness-tested across ordinary
moves, castling, en passant, promotion, king moves, null moves, and push/pop restoration. Its
whole-search benchmark was 20,153 NPS versus 23,580 NPS for full recomputation, a 14.5% regression.
Because this 32-unit network is unusually small, Python-side move bookkeeping costs more than the
compiled full evaluation saves. The architecture was therefore rejected before game testing.

## Selective-search experiments

Three targeted search changes were tested against the frozen incumbent on fresh, color-reversed
positions. None met the complete promotion gate:

- Shallow recapture extensions increased tactical tree size, lost completed depth in six of ten
  benchmark positions, and scored +6 =5 -9 (42.5%) over 20 fast games. Rejected.
- Countermove ordering preserved benchmark depth but scored +7 =6 -7 (50%) over 20 fast games.
  Rejected without confirmation.
- Conservative depth-one late-move futility pruning scored +8 =6 -6 (55%) in its pilot and
  +17 =8 -15 (52.5%) in a fresh 40-game confirmation. The combined fast result was 53.3%, but
  the required official-clock transfer check scored +1 =1 -2 (37.5%). Rejected rather than
  promoting a change that was not robust at the competition time control.

All three experiments had zero crashes, flags, illegal moves, or initialization failures. These
results reinforce that fast-clock screening is useful for rejecting weak ideas but is not enough
by itself to promote a search heuristic for 120+0.5 play.

## Wider and phase-bucketed evaluator experiments

The data splitter now hashes source game IDs by default, keeping every position from one game
entirely in either training or validation. On this stricter game-disjoint split, the handcrafted
evaluator scored 160.9 cp MAE.

- HalfKP64 reached 145.9 cp held-out MAE, but full inference fell to about 15,000 NPS and lost
  completed depth across most benchmark positions. It was rejected at the runtime gate before
  games.
- HalfKP32 with three material-phase output heads reached 145.3 cp held-out MAE while retaining
  about 24,700 NPS. Its fresh paired pilot scored +8 =4 -8 (50%) over 20 games, so it was rejected
  without confirmation.
- Reducing the proven residual blend from 50% to 33% scored +7 =4 -9 (45%) over 20 fresh games.
  The shipped 50% calibration remains unchanged.

Inference and training now support variable hidden widths and phase-bucketed output heads, making
these architecture experiments reproducible without locking the submission to an unproven model.
The shipped weight file still has one 32-unit output head and therefore retains identical scoring.

## Rapid seven-candidate search sweep

A new teacher-position screen compared seven search variants on 100 held-out MultiPV positions at
a small fixed clock before spending games. The incumbent scored 39% top-1 agreement, 64% top-3
agreement, and 20.0 cp mean regret where its move appeared in the teacher's top three.

- Reverse-futility margins of 70 and 120 cp per ply scored 38/64/20.6 and 40/66/16.4
  (top-1 percentage/top-3 percentage/regret cp), respectively.
- Safer and faster LMR variants scored 39/65/20.6 and 40/65/20.0.
- Requiring depth four for null-move pruning scored 39/64/19.8.
- Tightening quiescence SEE pruning to -80 cp scored 40/66/19.7.
- Widening the aspiration window to 50 cp regressed to 38/62/21.0 and was eliminated immediately.

The three screen leaders then played the same five reversed-color pairs against the incumbent:

- The 120 cp reverse-futility margin scored +3 =4 -3 (50%).
- The -80 cp quiescence SEE threshold scored +3 =3 -4 (45%).
- Faster LMR scored +4 =1 -5 (45%).

No candidate advanced. The result demonstrates that shallow teacher agreement is an efficient
rejection filter, not a substitute for games. All 30 game-stage trials completed without an agent
failure.

## Margin-stratified tactical gates

Raw top-move agreement mixes forced tactics with positions where several moves are effectively
equivalent. The rapid screen now accepts `--min-gap-cp`, defined as the Stockfish MultiPV score gap
between its first and second choices. On 100 held-out positions at the one-second-clock screen:

- At a 50 cp minimum gap, the incumbent scored 73% top-1 and 91% top-3.
- At a 100 cp minimum gap, it scored 84% top-1 and 93% top-3.
- At a 150 cp minimum gap, it scored 89% top-1 and 97% top-3.

This establishes an honest greater-than-80% metric for clear tactical decisions. It does not imply
an 80% game score or 80% agreement on ambiguous positions. On a separate 100-position unfiltered
sample, increasing the screen clock from 1,000 to 3,000 ms improved top-1 agreement from 35% to
42%, top-3 from 65% to 71%, and known top-three regret from 22.0 to 17.4 cp. Future candidates
should preserve at least 80% clear-tactic accuracy while improving ambiguous top-3 agreement and
paired game score.

### Deeper confirmation samples

Larger independent samples refined the initial estimates:

- 500 unfiltered positions at the short screen scored 38.8% top-1 (95% Wilson interval
  34.6--43.1%) and 64.8% top-3 (60.5--68.9%), with 24.1 cp known regret.
- 300 unfiltered positions at the deeper screen scored 40.7% top-1 (35.3--46.3%) and 65.7%
  top-3 (60.1--70.8%), with 16.2 cp known regret.
- 233 positions with a teacher gap of at least 100 cp scored 77.7% top-1 (71.9--82.6%) and
  88.0% top-3 (83.2--91.6%).
- 215 positions with a teacher gap of at least 150 cp scored 86.5% top-1 (81.3--90.4%) and
  94.4% top-3 (90.5--96.8%).

The strict tactical tier therefore exceeds 80% even at the lower confidence bound. The earlier
84% estimate for the 100 cp tier was optimistic sampling noise; its larger-sample point estimate
is 77.7%. Deeper search materially reduces regret on ambiguous positions but does not materially
change top-three agreement, indicating that evaluation/ranking is the next bottleneck.

## Fresh official-clock Stockfish-2600 sample

Five new color-reversed pairs were played from untouched positions at 120+0.5 against the local
Stockfish `UCI_LimitStrength=2600` wrapper. The agent scored +3 =3 -4 (45%), or -35 unanchored Elo,
with a pair-aware 95% interval of 35.2--54.8%. All ten games completed without a crash, flag,
illegal move, or initialization failure.

Combining this independently reported sample with the earlier 40 games gives +10 =16 -24 over 50
games: 36.0%, -100 unanchored Elo, and a pair-aware 95% score interval of 27.0--45.0% (-173 to -35
Elo). If the wrapper were perfectly calibrated at 2600, the point estimate would be about 2500;
the wrapper is an internal reference and not a certified public rating.

Stockfish analysis of the ten new PGNs found 32.1 cp competitive middlegame loss across 66 moves
and 30.0 cp competitive endgame loss across 197 moves. There were four competitive middlegame
mistakes above 100 cp, eleven competitive endgame mistakes above 100 cp, and eight mate-score
swings in endgames. Endgame conversion and late tactical stability are therefore the clearest
next targets.

## Upper-strength stress test

Against the local Stockfish 2700 configuration at the fast test clock, the promoted candidate
scored +2 =4 -14 over 20 games: 20%, approximately -241 unanchored Elo, pair-aware interval
5.8–34.2%. There were no operational failures. This configuration is not a calibrated rating
pool, but the result is strong evidence that the bot must not yet be represented as 2700 strength.

## Competition compliance

The competition documentation checked on 4 September permits team-trained model weights and
unrestricted training data, including third-party-engine annotations. The submission contains
only readable team code and the team-trained weight file. It contains no Stockfish executable,
wrapper, network access, subprocess engine call, opening leak, or hidden match information.

## Endgame-target experiments after the 2600 sample

Three narrowly scoped follow-ups were gated against the frozen incumbent:

- Adding quiet checks to the first two quiescence plies only in low-material positions was
  rejected at the runtime gate. Affected endgame throughput fell from roughly 30k to 20k nodes
  per second and representative positions lost one or two completed plies.
- A three-bucket model with the proven HalfKP feature transformer frozen and only its low-phase
  output head retrained reduced held-out residual error slightly. It scored +7 =6 -7 over 20
  paired fast games (50.0%, pair-aware 95% interval 42.7--57.3%) and was not promoted.
- Generational transposition-table retention scored +8 =3 -9 over 20 paired fast games (47.5%,
  pair-aware 95% interval 28.9--66.1%) and was not promoted. The short-clock test also did not
  establish that table-capacity events were frequent enough to isolate the mechanism.

The training utility now supports initializing from an existing float model, freezing its shared
feature transformer, and filtering by exact material phase. This makes future phase-head studies
cheap while preserving the incumbent representation. None of these experiments changed the
submitted engine or its weights.
