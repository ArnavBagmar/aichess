# Learned evaluator workflow

## HalfKP residual checkpoint (2026-09-04)

A 30,002-position realistic corpus was labelled offline by Stockfish 18 at depth 8. The first
HalfKP-32 network trained directly on teacher scores was rejected at 199.65 cp held-out MAE versus
163.18 cp for the handcrafted evaluator. Training the same two-perspective network on the teacher
minus handcrafted residual reached 144.98 cp, an 11.2% held-out improvement. This model is only a
candidate: quantized inference cost and paired-game strength must pass before its weights ship.

The corpus was recovered from an interrupted append-only run by retaining 30,002 valid unique JSON
records and rejecting one malformed line plus 532 duplicates. Its repair manifest records hashes
and counts. `tools/label_positions.py --resume` now makes future labeling runs checkpoint-safe.

The learned evaluator is an optional, measured replacement layer—not a reason to weaken the
working alpha-beta engine. Stockfish runs only offline as a teacher. Neither its executable nor its
weights enter `submission.zip`.

## 1. Create labels

```powershell
python -m tools.label_positions `
  --stockfish C:\path\to\stockfish.exe `
  --output data\sf18-d10-v1.jsonl `
  --positions 100000 --depth 10 --max-abs-score 1000 --multipv 3
```

For move-policy work, `--all-legal-moves` dynamically sets MultiPV to the exact legal root-move
count and verifies that every move received a score. This is an offline-only, high-cost labeling
mode; the teacher executable is never packaged.

The generator uses one teacher thread, fixed-seed legal randomized games, score filtering, and
records several teacher-ranked legal moves plus their scores. It writes a SHA-256 provenance
manifest. The JSONL dataset and its machine-specific manifest are
ignored by Git. For a production dataset, record the Stockfish release, command line, generator
commit, seed, filtering, and dataset hash in the final model's provenance.

## 2. Train the residual evaluator

```powershell
python -m tools.train_evaluator data\sf18-d10-v1.jsonl `
  --output weights\evaluator.json --epochs 100 --patience 8
```

The 768 integer parameters represent middle-game and endgame piece-square corrections. Training
starts from the current handcrafted evaluation and learns only a residual. MultiPV child positions
teach the model about score differences between actual legal choices. A deterministic 10%
validation split selects the checkpoint; early stopping prevents exporting a later overfit epoch.
The exported JSON embeds architecture, dataset hash, hyperparameters, seed, and held-out error.

## 3. Promotion gate

A model may affect `agent.py` only after all of these pass:

1. Dataset provenance is complete and the weights were trained by this team.
2. Held-out MAE and legal-move ranking improve over the handcrafted baseline.
3. Evaluation throughput and search depth do not regress materially on one pinned CPU core.
4. Paired games beat the no-model checkpoint across varied FENs, followed by a Stockfish-level
   calibration match.
5. Low-clock, board-restoration, mate, draw, and package tests pass.
6. `submission.zip` contains only readable engine source and the promoted team-trained artifact,
   remains below 50 MB unzipped, and performs no runtime network or subprocess activity.

The 250-position smoke run intentionally was not promoted. It validated the pipeline, but its
held-out improvement was only about 3.5 centipawns on roughly 25 validation examples—far too small
and noisy to claim playing-strength evidence.

The 5,000-position depth-10 pilot also was not promoted. Its residual reduced held-out MAE from
309.1 to 265.2 centipawns (14.2%) with effectively unchanged search throughput, but scored only
5 wins, 1 draw, and 6 losses in a 12-game reversed-color match against the handcrafted checkpoint
(-29 unanchored Elo). This is evidence that teacher-score regression alone is not our objective.
The next dataset/model iteration must emphasize quiet legal game positions and pairwise legal-move
ranking, then clear the same paired-game gate.

A subsequent real-game pilot sampled 2,000 balanced positions from a deterministic 5,000-position
Lichess reservoir and labelled three Stockfish 18 depth-10 variations per root. The residual reduced
held-out MAE from 172.2 to 166.1 centipawns (3.5%). That was below the promotion threshold, so its
weights were deleted. The result supports richer interaction features or a compact nonlinear model
rather than further scaling a piece-square-only residual.

## WDL and legal-move ranking experiment (2026-09-05)

`tools/train_ranked_nnue.py` adds logistic WDL loss and pairwise logistic ranking between the
teacher's MultiPV moves to an initialized HalfKP model. Child positions use the correctly negated
side-to-move perspective, and WDL targets can blend teacher expectation with the source result.

The first 2,000-position depth-10 MultiPV-3 experiment moved held-out top-1 agreement from 48.2%
to 49.2%, but its 20-game paired pilot scored +4 =7 -9 (37.5%, about -89 unanchored Elo). It was
rejected and the incumbent weights were restored.

The labeler then gained a `--max-best-gap` filter and produced 5,000 balanced depth-12 MultiPV-3
positions restricted to a best/second-best gap of at most 150 cp. The run required 5,776
Stockfish 18 calls and produced 15,000 ranked moves. Its manifest records dataset SHA-256
`6a99ce2956860d43c18266b74616492160783585c5a72de3fa7d4a67663aacff`.

Across three predeclared objective mixes, WDL log-loss improved slightly but the best held-out
top-1 agreement was only 43.5% versus the initialized model's 43.3%. Low-rate shared-feature
fine-tuning also stayed flat or regressed. No depth-12 model advanced to games. This indicates that
a scalar value head is an inadequate move-policy surrogate on close choices. The next ranking
architecture should learn explicit move features and be tested as a low-cost ordering signal,
without replacing the proven leaf evaluator.

### Explicit quiet-move policy follow-up

A 1,749-parameter phase-aware linear policy was trained directly on quiet MultiPV pairs. It used
moving-piece, oriented source, oriented destination, and piece-destination features. The best run
reached 55.8% held-out pairwise accuracy among deliberately close quiet choices.

Applying the policy at every node reduced searched node counts but cost about 11% raw NPS and lost
completed depth on two benchmark positions. Precomputing all additive terms into one lookup and
limiting policy use to the first two plies removed most overhead, but the paired-game pilot still
scored only +6 =6 -8 over 20 games (45.0%, about -35 unanchored Elo). The policy and candidate
weights were rejected; production code and weights were restored.

The result establishes a useful lower bound: a move-ordering classifier near 56% pairwise accuracy
is not strong enough to interact safely with this engine's LMR and pruning. Future policy work
should require materially higher held-out accuracy, include negative examples beyond teacher top-3,
and be evaluated on node reduction at equal completed score/depth before any games.

### Complete-legal-move policy follow-up (2026-09-05)

The labeler produced 5,000 realistic, close-choice positions at Stockfish 18 depth 8, scoring every
legal root move. It accepted 5,000 of 5,533 unique positions under a 200 cp best/second-best filter;
the dataset SHA-256 is
`7d0d28f35638c61cdea85c0c8e9b8fafdce07be5edf4c7ed50f5dc7536a8a56e`.

The second policy uses a 65,536-entry hashed linear model with twelve active features per move:
phase, piece/source/destination interactions, move geometry, captures, promotions, checks, and
destination attack/defense context. Game-disjoint validation on 527 positions reached a best 66.9%
broad pairwise accuracy, but only 57.3% on move pairs separated by at most 100 cp. Top-1 agreement
remained 19% or lower and mean selected-move regret remained roughly 141--145 cp.

This candidate was rejected before runtime or games. The broad 65% threshold alone is too easy to
satisfy with obviously inferior all-move negatives; hard-pair accuracy, top-choice agreement, and
regret show that the signal is still insufficient for pruning-sensitive move ordering. Production
`agent.py` and `nnue.npz` were not changed. A future iteration needs a board-aware policy, not
further tuning of this linear feature family.
