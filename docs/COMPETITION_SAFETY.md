# Competition-safe design policy

This policy was checked against the live [agent documentation](https://aichessathon.com/docs) and
[competition rules](https://aichessathon.com/terms) on 3 September 2026. Re-check both before every
upload. If documents conflict, the latest dated competition rules take precedence.

## Approved architecture

Our recommended design is competition-safe:

- team-written iterative-deepening alpha-beta/PVS search;
- team-written evaluation, move ordering, pruning, time management, and transposition storage;
- `python-chess`, NumPy, Numba, PyTorch CPU, or ONNX Runtime from the fixed platform environment;
- a compact evaluator whose weights were trained by the team;
- offline training positions or labels produced by Stockfish or another engine;
- legally sourced opening books and Syzygy files shipped as data;
- module state, transposition reuse, and pondering within the process for one game.

Published algorithms are ideas we may independently implement. Surveyed engine repositories are
architecture references, not sources to port into the submission.

## Hard prohibitions

The submission must not contain or invoke:

- Stockfish, Lc0, Maia, or any other third-party chess engine;
- a wrapper, package, executable, subprocess, or service that selects moves using such an engine;
- pretrained third-party engine weights presented as team-trained weights;
- native binaries, compiled extensions, shared libraries, or locally supplied wheels;
- network access, hosted inference, APIs, downloads, or telemetry;
- obfuscated source or intentionally opaque move-selection logic;
- attempts to access other participants, hidden match data, organizer systems, or credentials;
- dependencies beyond the standard library and the five packages in the live documentation.

Do not use competition-engine assistance for Daily Five. The agent competition permits offline
engine-labelled training data; Daily Five has separate human-only fair-play rules.

## Design gates

### Search and board state

- Return only a legal UCI move from the supplied FEN.
- Select a legal fallback before starting timed work.
- Treat the FEN side to move as our color; assume no other private input.
- Use monotonic wall-clock deadlines with a hard safety reserve.
- Keep all caches bounded below 2 GB, including Python overhead.
- Make state reusable only within one process/game and safe when a new or unexpected FEN arrives.
- Pondering is allowed, but it must not prevent the next call from meeting its deadline.

### Learned evaluation

- Train every shipped model ourselves.
- Record dataset sources, filters, teacher version, training code revision, hyperparameters, seed,
  metrics, checkpoint hash, export command, and final artifact hash.
- Ensure the learned model materially participates if we describe the entry as model-driven.
- Never load a teacher engine or teacher network during competition execution.
- Export only supported weight formats and test CPU-only inference without network access.
- Count weights against the 50 MB unzipped package limit.

### Books and tablebases

- Record source, license, generation method, file hashes, and exact included material classes.
- Ship them only as data and use the platform's `chess.polyglot` or `chess.syzygy` support.
- Count every file uncompressed against 50 MB.
- Do not disguise an existing engine or opaque move-selection database as a book.
- Demonstrate that our own search remains the general move-selection mechanism.

### Numba and performance

- Numba JIT compilation in process is allowed; warm required kernels during the 60-second init.
- Do not ship Numba caches or compiled artifacts.
- Use one active compute thread during our move unless measurement proves otherwise; all threads
  share the single dedicated core.
- If using PyTorch, constrain its intra-op and inter-op thread counts to one.
- Keep initialization below 60 seconds on a cold, platform-like machine.

### Filesystem and process behavior

- Treat the submission directory as read-only.
- Write only disposable files below `/tmp`, within the 256 MB scratch allowance.
- Do not assume a normal home directory or persistent files across games.
- Do not launch external engine processes or depend on shell tools.
- Avoid filenames such as `chess.py`, `types.py`, `random.py`, or others that shadow imports.
- Keep diagnostic output small; a move response over 4096 bytes is a loss.

## Provenance record

Maintain a private or repository-visible build manifest for each candidate submission:

```text
source_commit:
rules_checked_at:
rules_version:
training_code_commit:
teacher_engine_and_version:
dataset_sources_and_licenses:
dataset_manifest_hash:
training_configuration_hash:
model_artifact_hash:
book_or_tablebase_sources:
package_file_list:
package_unzipped_bytes:
platform_smoke_result:
```

This makes the final walkthrough straightforward and provides evidence that the model and engine are
ours even though public data and offline teacher labels informed training.

## Pre-upload compliance gate

Every release candidate must pass all of the following:

1. Re-read the live documentation and dated rules; record their version/date.
2. Review the diff from the last compliant submission.
3. Inspect the complete zip file list—no accidental datasets, secrets, caches, binaries, or tools.
4. Verify `agent.py` is at zip root and imports in a fresh process.
5. Verify total **unzipped** size is at most 50 MB.
6. Scan imports against the standard library plus the five permitted packages.
7. Run without network and with the submission directory read-only.
8. Run on Linux, Python 3.12, one core, and a 2 GB memory limit.
9. Measure cold initialization below 60 seconds.
10. Complete smoke games as both colors without crash, illegal move, or flag.
11. Test very low clocks and retain an always-legal fallback.
12. Confirm model, book, tablebase, and dataset provenance records are complete.
13. Build the final zip from a clean commit and record its cryptographic hash.
14. Read the platform validation log; platform acceptance is the final operational authority.

## Safe interpretation of specific ideas

| Idea | Status | Boundary |
|---|---|---|
| Alpha-beta, PVS, TT, LMR, null move, quiescence | Safe | Implement independently |
| Study Stockfish source and documentation | Safe | Do not port or ship it |
| Use Stockfish labels offline | Explicitly allowed | Teacher absent from submission |
| Team-trained NNUE-style model | Safe | Our data pipeline, training, and weights |
| ONNX, `.pt`, or `.safetensors` weights | Safe | Team-trained and within size/runtime limits |
| Numba JIT | Safe | Compile in process; no shipped binary cache |
| Opening book or Syzygy data | Explicitly allowed | Legal provenance and within 50 MB |
| Pondering between calls | Explicitly allowed | One-core and deadline safe |
| Opponent analysis from public games | Safe | No hidden or unauthorized access |
| Existing engine package or executable | Prohibited | Includes wrappers and indirect invocation |
| Runtime download or hosted model | Impossible/prohibited | Network is disabled |

When a future design does not fit clearly into this table, pause implementation and obtain a written
organizer clarification at the contact address in the official documentation.
