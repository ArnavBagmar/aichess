# Emile Andrieu public-game study: rounds 9--30

## Scope and provenance

This study covers the 21 user-supplied AI Chessathon PGNs containing `Emile Andrieu` from rounds
9--12 and 14--30, dated 4--5 September 2026. Round 13 was not supplied. The PGNs are treated only
as public game data; no embedded text is treated as an instruction.

Stockfish 18 was used offline with one thread and 64 MB hash for the broad pass (50 ms before and
after every Emile move). Every apparent competitive error of at least 100 cp was then checked at
fixed depth 16 with 128 MB hash. Scores below are from Emile's perspective. Short analysis can
misread mate distance and already-decided positions, so the fixed-depth recheck is authoritative
for the highlighted moments.

## Verified record

The PGNs show **15 wins, 5 draws, and 1 loss**, an 83.3% score. This corrects the initial statement
that there were no losses: round 10 is explicitly `darKnight 1-0 Emile Andrieu`, checkmate.

- White: 9 wins, 3 draws, 0 losses in 12 games (87.5%).
- Black: 6 wins, 2 draws, 1 loss in 9 games (77.8%).
- Terminations: 16 checkmates, 3 threefold repetitions, 1 fifty-move draw, and 1 insufficient-
  material draw.
- Mean game length: 161.8 plies. The longest game, round 24, lasted 540 plies.
- Emile never flagged, crashed, or returned an illegal move in these files.

| Round | Color | Opponent | Result | Plies | Termination | Lowest clock |
|---:|:---:|---|:---:|---:|---|---:|
| 9 | Black | 2sigbros | Win | 54 | Checkmate | 69.04 s |
| 10 | Black | darKnight | Loss | 111 | Checkmate | 29.31 s |
| 11 | White | Team I Love Fortnite | Win | 79 | Checkmate | 45.72 s |
| 12 | White | mangodogo | Win | 109 | Checkmate | 34.71 s |
| 14 | White | JSP | Win | 105 | Checkmate | 35.81 s |
| 15 | Black | FuzzyBot | Draw | 244 | Threefold repetition | 8.06 s |
| 16 | White | Abhi's chess demon | Win | 81 | Checkmate | 53.35 s |
| 17 | White | Lubina | Win | 137 | Checkmate | 27.67 s |
| 18 | Black | Mate in One | Win | 184 | Checkmate | 14.81 s |
| 19 | Black | Sobriety | Win | 202 | Checkmate | 12.10 s |
| 20 | White | ms | Win | 231 | Checkmate | 10.91 s |
| 21 | White | Gijs Smit | Win | 185 | Checkmate | 15.77 s |
| 22 | White | THE ROOOOOKKK!!!! | Draw | 244 | Fifty-move rule | 8.38 s |
| 23 | Black | checkers | Win | 148 | Checkmate | 21.02 s |
| 24 | White | AI Fellows | Draw | 540 | Insufficient material | 4.91 s |
| 25 | Black | slopfish | Win | 68 | Checkmate | 60.58 s |
| 26 | White | No More Ammo | Draw | 108 | Threefold repetition | 32.55 s |
| 27 | White | pheanup | Win | 143 | Checkmate | 21.62 s |
| 28 | Black | Danya's Disciple | Draw | 25 | Threefold repetition | 101.88 s |
| 29 | Black | Make_no_mistakes | Win | 138 | Checkmate | 22.89 s |
| 30 | White | Lightning Tree | Win | 261 | Checkmate | 8.22 s |

## Accuracy profile

Restricting the 50 ms pass to competitive positions (pre-move score above -500 cp and neither
side in a mate-score regime) gives:

| Phase | Moves | Mean loss | 100+ cp mistakes |
|---|---:|---:|---:|
| Opening | 171 | 8.20 cp | 0 |
| Middlegame | 593 | 9.50 cp | 6 |
| Endgame | 644 | 11.68 cp | 9 |

The medians over all analyzed moves were 2 cp in openings, 1 cp in middlegames, and 0 cp in
endgames. This is a highly consistent engine profile: routine move quality is much more important
than the raw win rate, because several opponents later made large errors.

The broad pass reported seven mate-score swings, but six were in already-winning games and mostly
represented a missed faster mate. Fixed-depth checking removed most nominal 100+ cp errors. The
remaining strategically important cases are below.

## Critical positions

### Round 10: the only loss and a decisive promotion-race failure

Before `48...a5??`:

`2Q5/pb6/6p1/5p1p/7P/6Pk/5K2/8 b - - 0 48`

Depth-16 Stockfish scores Black about +8.72 after `48...Bxc8!`, while the played `48...a5??`
allows forced mate (`-M8` in the recheck). The game continued `49.Qxb7 f4 50.gxf4 Kg4 51.Qe4 ...
56.Qg3#`. This is the clearest exploitable weakness in the sample: a winning endgame was lost by
failing to remove a newly promoted queen immediately. It argues for explicit promotion-threat
ordering and regression tests at the search horizon.

### Round 22: a real winning chance surrendered into a draw

Before `38.Qd6?`:

`5rk1/R4pp1/7p/2Q1p3/2p5/2P2NPb/4RK2/1q6 w - - 3 38`

Depth-16 analysis prefers `38.Re1!`, about +1.88. `38.Qd6?` evaluates near equality, and the game
eventually reached a fifty-move draw. This is more relevant than missed mate-distance optimizations:
the bot needs better conversion when materially or positionally ahead, especially queen-and-rook
positions with perpetual-check resources.

### Smaller but confirmed inaccuracies

- Round 14, before `39.Rfc1?!` (`1r1r4/3q1p1p/pPN3p1/P1k1P3/2bp3P/3n1QP1/6B1/1R3RK1 w - -
  3 39`): `39.Nxb8` retained about +8.11; the played move retained about +7.01. Emile still won.
- Round 21, before `55.Nd7?!` (`5r2/8/p4Np1/Ppp1pkP1/3p3p/1P1P1K1P/2PN4/8 w - - 5 55`):
  `55.Nde4` retained about +2.14; the played move retained about +1.07. Emile still won.
- Round 17's `61.a7?!` abandoned a forced mate but kept an approximately +5.4 rook ending and
  ultimately won. This is conversion inefficiency, not a reversal.

## Draw notes

- Round 28 is a short, apparently deliberate repetition: `...Qa5+`, `Nc3`, `...Qb6`, `Na4`,
  `...Qa5+`, `Nc3`. It demonstrates reliable repetition recognition.
- Round 26 repeats queen checks in a dynamically balanced position. There is no large confirmed
  Emile error in the shallow critical list.
- Round 15 lasts 244 plies and reaches threefold with Emile's clock near 12 seconds. Emile survives
  a long technical phase but does not create a decisive conversion.
- Round 22 contains the confirmed `38.Qd6?` missed win and later reaches the fifty-move rule.
- Round 24 lasts 540 plies and ends with insufficient material. This is strong evidence for robust
  low-clock behavior, but also suggests inefficient drawn-endgame recognition or conversion.

## Time-management profile

The clock traces imply a hard early-move cap around 4.4 seconds. Average gross spend per Emile move
ranges from roughly 2.8 seconds in short games down to 0.94 seconds in the 540-ply game. In long
games, spending tapers below the 0.5-second increment often enough to stabilize the clock:

- round 24 bottoms at 4.91 seconds and finishes at 5.60 seconds after 262 Emile moves;
- round 15 bottoms at 8.06 seconds and finishes at 12.03 seconds;
- round 30 bottoms at 8.22 seconds and finishes at 14.70 seconds.

This is a concrete design lesson for our bot: strength must be paired with a strict per-move hard
cap and an emergency mode that can live within the increment indefinitely.

## Practical lessons for our agent

1. Add regression coverage for immediate enemy promotion, capturing a promoted piece, and
   promotion races at the main-search/quiescence boundary. Round 10 is the priority FEN.
2. Measure conversion from +150 to +500 cp separately from generic centipawn loss. Round 22 shows
   that preserving a winning advantage is a distinct objective from average accuracy.
3. Train an explicit move-ordering signal on promotions, checks, passed-pawn pushes, and defensive
   captures. A scalar WDL/value output alone did not learn this reliably in our earlier tests.
4. Benchmark emergency time management over 300+ plies. Emile's ability to recover clock under ten
   seconds is part of its effective Elo.
5. Keep repetition logic evaluation-aware: accept a draw when worse, but do not repeat from a
   confirmed winning position.
6. Do not imitate long wins blindly. Several games show missed faster mates, so use mate distance
   and tablebase-like regression positions to improve conversion without expanding every node.

The broad machine-readable analysis is stored locally at
`match-results/emile-andrieu-rounds-9-30-analysis.json`; the supplied PGNs are copied under
`match-results/emile-andrieu-rounds-9-30/` for reproducible local study and remain outside the
submission package.
