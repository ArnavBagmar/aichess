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
