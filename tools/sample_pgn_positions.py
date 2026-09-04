"""Sample reproducible, realistic positions from a streaming PGN archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import chess.pgn


def integer_header(headers: chess.pgn.Headers, name: str) -> int:
    try:
        return int(headers.get(name, "0").rstrip("?"))
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--max-ply", type=int, default=100)
    parser.add_argument("--min-elo", type=int, default=1_800)
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reservoir: list[dict[str, object]] = []
    eligible = games_read = 0
    source_digest = hashlib.sha256()
    with args.pgn.open("rb") as source_bytes:
        for block in iter(lambda: source_bytes.read(1024 * 1024), b""):
            source_digest.update(block)

    with args.pgn.open(encoding="utf-8") as source:
        while games_read < args.max_games:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            games_read += 1
            white_elo = integer_header(game.headers, "WhiteElo")
            black_elo = integer_header(game.headers, "BlackElo")
            if min(white_elo, black_elo) < args.min_elo:
                continue
            moves = list(game.mainline_moves())
            upper = min(args.max_ply, len(moves) - 1)
            if upper < args.min_ply:
                continue
            target = rng.randint(args.min_ply, upper)
            board = game.board()
            for move in moves[:target]:
                board.push(move)
            if board.is_game_over() or not board.is_valid():
                continue
            eligible += 1
            record: dict[str, object] = {
                "fen": board.fen(en_passant="fen"),
                "game_id": game.headers.get("Site", ""),
                "ply": target,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": game.headers.get("Result", "*"),
            }
            if len(reservoir) < args.positions:
                reservoir.append(record)
            else:
                replacement = rng.randrange(eligible)
                if replacement < args.positions:
                    reservoir[replacement] = record
            if games_read % 10_000 == 0:
                print(f"read {games_read} games; {eligible} eligible")

    rng.shuffle(reservoir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in reservoir
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "format": 1,
        "generator": "tools/sample_pgn_positions.py",
        "source": args.pgn.name,
        "source_sha256": source_digest.hexdigest(),
        "seed": args.seed,
        "games_read": games_read,
        "eligible_games": eligible,
        "positions": len(reservoir),
        "min_elo": args.min_elo,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
