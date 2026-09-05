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
