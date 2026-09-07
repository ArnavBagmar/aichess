"""Extract positions faced by one player from public PGNs into JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.pgn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_dir", type=Path)
    parser.add_argument("--player", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for path in sorted(args.pgn_dir.glob("*.pgn")):
        with path.open(encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            continue
        if game.headers.get("White") == args.player:
            color = chess.WHITE
        elif game.headers.get("Black") == args.player:
            color = chess.BLACK
        else:
            continue
        board = game.board()
        player_index = 0
        for node in game.mainline():
            move = node.move
            if board.turn == color:
                if player_index % args.stride == 0:
                    records.append(
                        {
                            "fen": board.fen(en_passant="fen"),
                            "played_move": move.uci(),
                            "game_id": path.stem,
                            "round": game.headers.get("Round"),
                            "result": game.headers.get("Result"),
                            "clock_seconds": node.clock(),
                        }
                    )
                player_index += 1
            board.push(move)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {len(records)} positions to {args.output}")


if __name__ == "__main__":
    main()
