"""Generate diverse legal positions and label them with an offline UCI teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import TextIO, TypedDict, cast

import chess


class Candidate(TypedDict):
    rank: int
    move: str
    score_cp: int


class UciTeacher:
    """Minimal offline-only UCI client; this module is never included in submissions."""

    def __init__(self, executable: Path, hash_mb: int, multipv: int) -> None:
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open UCI teacher pipes")
        self.stdin = cast(TextIO, self.process.stdin)
        self.stdout = cast(TextIO, self.process.stdout)
        self.name = "unknown UCI teacher"
        self._send("uci")
        for line in self.stdout:
            if line.startswith("id name "):
                self.name = line.removeprefix("id name ").strip()
            if line.strip() == "uciok":
                break
        else:
            raise RuntimeError("teacher exited before uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, command: str) -> None:
        self.stdin.write(command + "\n")
        self.stdin.flush()

    def _wait_for(self, token: str) -> None:
        for line in self.stdout:
            if line.strip() == token:
                return
        raise RuntimeError(f"teacher exited before {token}")

    def analyze(
        self, board: chess.Board, depth: int
    ) -> tuple[int, str, int, list[Candidate]]:
        self._send(f"position fen {board.fen()}")
        self._send(f"go depth {depth}")
        score = 0
        nodes = 0
        bestmove = "0000"
        candidates: dict[int, tuple[int, str]] = {}
        for line in self.stdout:
            fields = line.split()
            if fields[:1] == ["info"] and "score" in fields:
                index = fields.index("score")
                if fields[index + 1] == "cp":
                    score = max(-2_000, min(2_000, int(fields[index + 2])))
                elif fields[index + 1] == "mate":
                    score = 2_000 if int(fields[index + 2]) > 0 else -2_000
                if "nodes" in fields:
                    nodes = int(fields[fields.index("nodes") + 1])
                rank = int(fields[fields.index("multipv") + 1]) if "multipv" in fields else 1
                if "pv" in fields:
                    candidates[rank] = (score, fields[fields.index("pv") + 1])
            if fields[:1] == ["bestmove"]:
                bestmove = fields[1]
                ranked: list[Candidate] = [
                    {"rank": rank, "move": move, "score_cp": candidate_score}
                    for rank, (candidate_score, move) in sorted(candidates.items())
                ]
                principal_score = ranked[0]["score_cp"] if ranked else score
                return principal_score, bestmove, nodes, ranked
        raise RuntimeError("teacher exited during analysis")

    def close(self) -> None:
        if self.process.poll() is None:
            self._send("quit")
            self.process.wait(timeout=5)

    def __enter__(self) -> UciTeacher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def random_position(rng: random.Random, minimum_ply: int, maximum_ply: int) -> chess.Board:
    """Sample a legal position from a randomized game, avoiding terminal boards."""
    board = chess.Board()
    target = rng.randint(minimum_ply, maximum_ply)
    for _ in range(target):
        moves = list(board.legal_moves)
        if not moves:
            return random_position(rng, minimum_ply, maximum_ply)
        # Mild capture/check preference makes useful tactics less vanishingly rare.
        forcing = [move for move in moves if board.is_capture(move) or board.gives_check(move)]
        move = rng.choice(forcing if forcing and rng.random() < 0.35 else moves)
        board.push(move)
    if board.is_game_over():
        return random_position(rng, minimum_ply, maximum_ply)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=100)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--max-abs-score", type=int, default=1_000)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to a valid partial output, skipping FENs already written",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_records: list[dict[str, object]] = []
    if args.input_jsonl:
        input_records = [
            json.loads(line)
            for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        ]
        rng.shuffle(input_records)
    input_index = 0
    seen: set[str] = set()
    attempted = 0
    accepted = 0
    resumed_positions = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if args.resume and args.output.exists():
        for line_number, line in enumerate(
            args.output.read_text(encoding="utf-8").splitlines(keepends=True), 1
        ):
            try:
                record = json.loads(line)
                seen.add(str(record["fen"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise SystemExit(
                    f"cannot resume: invalid record at {args.output}:{line_number}"
                ) from error
            digest.update(line.encode())
            accepted += 1
        resumed_positions = accepted
        print(f"resuming from {accepted} valid positions")
    output_mode = "a" if args.resume else "w"
    with (
        args.output.open(output_mode, encoding="utf-8") as destination,
        UciTeacher(args.stockfish.resolve(), args.hash_mb, args.multipv) as teacher,
    ):
        while accepted < args.positions:
            source_record: dict[str, object] = {}
            if input_records:
                if input_index >= len(input_records):
                    break
                source_record = input_records[input_index]
                input_index += 1
                board = chess.Board(str(source_record["fen"]))
            else:
                board = random_position(rng, args.min_ply, args.max_ply)
            key = board.fen(en_passant="fen")
            if key in seen:
                continue
            seen.add(key)
            score, bestmove, nodes, candidates = teacher.analyze(board, args.depth)
            attempted += 1
            if abs(score) > args.max_abs_score:
                continue
            record = {
                "fen": key,
                "score_cp": score,
                "bestmove": bestmove,
                "teacher_nodes": nodes,
                "candidates": candidates,
                "source": source_record,
            }
            line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            destination.write(line)
            digest.update(line.encode())
            accepted += 1
            if accepted % 100 == 0:
                print(f"labelled {accepted}/{args.positions} ({attempted} examined)")

    manifest = {
        "format": 1,
        "generator": "tools/label_positions.py",
        "teacher": teacher.name,
        "depth": args.depth,
        "seed": args.seed,
        "positions_examined": len(seen),
        "positions_accepted": accepted,
        "resumed_positions": resumed_positions,
        "teacher_calls": attempted,
        "max_abs_score": args.max_abs_score,
        "multipv": args.multipv,
        "input": args.input_jsonl.name if args.input_jsonl else "synthetic-random-play",
        "input_sha256": (
            hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()
            if args.input_jsonl
            else None
        ),
        "sha256": digest.hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
