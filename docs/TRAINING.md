# Learned evaluator workflow

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
