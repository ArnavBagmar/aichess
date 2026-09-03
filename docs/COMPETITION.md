# Competition brief

This note records our understanding of the public AI Chessathon site as of 2 September 2026.
The live [agent documentation](https://aichessathon.com/docs) and dated
[competition rules](https://aichessathon.com/terms) remain authoritative and should be checked
again before every upload.

## Premise

AI Chessathon is a chess-engine engineering competition. Teams write a self-contained Python
agent; the platform supplies a FEN position and remaining clock time, and the agent returns one
legal move in UCI notation. The platform runs submitted agents against one another on identical
hardware. There is no presentation score: chess strength, time discipline, legality, and runtime
reliability determine the results.

A model is optional. Classical search, a team-trained model, or a hybrid are valid approaches.
The practical problem is to produce the most playing strength possible on one CPU core while
never flagging, crashing, or returning an illegal move.

## Event flow

1. Register alone or in a team of up to three.
2. From 4-11 September, the latest validated build plays hourly rated ladder rounds from 08:00
   through 22:00. The public Elo ladder seeds, but does not decide, final qualification.
3. Uploads close at 11:00 on 11 September. The last validated builds are locked for a 13-round
   Swiss. Standings use points, Buchholz, head-to-head, then earlier final submission.
4. The Swiss determines the order in which 50 seats are offered for the London final.
5. The seated teams play an in-person knockout on 12 September at Encode Club, London.

Entry and the online ladder are open worldwide. A team enters the final qualification stage if at
least one member is a UK university student, and only eligible UK members may take London seats.
The organisers verify eligibility. The advertised prizes are GBP 1,000 for first, GBP 500 for
second, and GBP 250 for third.

## Daily Five

Daily Five is a separate individual chess-solving route from 6-10 September. A signed-in player
gets five assigned positions and 20 minutes; one wrong move ends a position. Engine assistance,
other people, position lookup, and additional accounts are forbidden. The top three eligible
players per day receive non-bracket Finals Day Wildcards, limited to one per person. This activity
does not affect an agent's Elo or Swiss result.

## Submission contract

- A zip no larger than 50 MB expanded, with `agent.py` at its root.
- `get_move(fen: str, time_left_ms: int) -> str` returns a legal UCI move.
- Python 3.12 with torch 2.13 CPU, numpy 2.5.2, python-chess 1.11.2, onnxruntime 1.29, and
  numba 0.67. Nothing else is installed during validation.
- One dedicated CPU core, 2 GB RAM, no GPU, no network, and a read-only filesystem except for
  256 MB under `/tmp`.
- A fresh process per game, a 60-second initialization allowance, and persistent module state
  between moves in that game. Pondering is allowed.
- 120 seconds per side plus 0.5 seconds per move. FIDE draws are claimed automatically. At 300
  plies an unfinished game is adjudicated by material, otherwise drawn.
- Rated games start from curated positions near equality, not necessarily the initial position.
- Six uploads per team per day; the latest version that passes two smoke games is active.

Illegal or malformed output, more than 4 KB of move output, a crash, out-of-memory termination,
initialization timeout, or flag fall loses the game.

## Clean-room boundary

Stockfish, Leela Chess Zero, Maia, and wrappers around existing engines may not ship in or select
moves for the submission. Native binaries and obfuscated source are also prohibited. A submitted
model must be trained by the team.

The rules explicitly permit unrestricted training data, including positions annotated by an
existing engine. Opening books and tablebases may ship as data. We can therefore use Stockfish as
an offline teacher and study published algorithms, but the submitted search, evaluation,
integration, and model must be our own readable work.

Participants retain ownership of their original code. Entry gives the organisers limited rights
to run, test, review, and record it. The public archive will eventually preserve rules, match data,
results, and approved participant material.

## Strategic consequences

- Arbitrary-FEN strength matters more than a move-one opening book.
- Search throughput and evaluator quality must be balanced; either one can bottleneck strength.
- Iterative deepening and clock checks are correctness features, not optional optimizations.
- Single-threaded CPU inference favors a compact evaluator over a large policy model.
- State retained across moves can support repetition awareness, transposition reuse, and pondering.
- Validation and packaging are part of playing strength because operational failures are losses.

