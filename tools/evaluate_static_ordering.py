"""Measure one-ply handcrafted/NNUE evaluation as a legal-move ordering policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from agent import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    correct = pairs = hard_correct = hard_pairs = top_one = count = 0
    regret = 0.0
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        source = record.get("source")
        game_id = source.get("game_id") if isinstance(source, dict) else None
        split_key = str(game_id) if game_id else record["fen"]
        digest = hashlib.blake2b(split_key.encode(), digest_size=2).digest()
        validation = int.from_bytes(digest, "little") % 10 == 0
        if args.validation_only and not validation:
            continue
        board = chess.Board(record["fen"])
        predictions: list[float] = []
        teacher_scores: list[float] = []
        for candidate in record.get("candidates", ()):
            move = chess.Move.from_uci(candidate["move"])
            if move not in board.legal_moves:
                continue
            board.push(move)
            try:
                predictions.append(float(-evaluate(board)))
            finally:
                board.pop()
            teacher_scores.append(float(candidate["score_cp"]))
        if len(predictions) < 2:
            continue
        predicted = np.asarray(predictions)
        scores = np.asarray(teacher_scores)
        selected = int(np.argmax(predicted))
        best = int(np.argmax(scores))
        top_one += int(selected == best)
        regret += float(scores[best] - scores[selected])
        order = np.argsort(scores)[::-1]
        ordered_predictions = predicted[order]
        ordered_scores = scores[order]
        pair_mask = np.triu(np.ones((len(order), len(order)), dtype=bool), k=1)
        wins = ordered_predictions[:, None] > ordered_predictions[None, :]
        hard_mask = pair_mask & (ordered_scores[:, None] - ordered_scores[None, :] <= 100)
        correct += int(np.count_nonzero(wins & pair_mask))
        pairs += int(np.count_nonzero(pair_mask))
        hard_correct += int(np.count_nonzero(wins & hard_mask))
        hard_pairs += int(np.count_nonzero(hard_mask))
        count += 1
    print(
        f"positions {count}, pair {correct / pairs:.1%}, "
        f"hard-pair {hard_correct / hard_pairs:.1%}, top-1 {top_one / count:.1%}, "
        f"regret {regret / count:.2f} cp"
    )


if __name__ == "__main__":
    main()
