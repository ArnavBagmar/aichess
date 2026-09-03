"""A test-only, approximately 1000-strength opponent.

This is a project-authored sparring profile, not an independently calibrated Elo claim. It uses
two-ply material/placement search and controlled move noise to emulate a novice club opponent.
Nothing in this directory is included by the submission packager.
"""

import math
import random

import chess

VALUES = (0, 100, 310, 325, 500, 900, 0)
MATE = 100_000
RNG = random.Random(1000)


def evaluate(board: chess.Board, perspective: chess.Color) -> int:
    if board.is_checkmate():
        return -MATE if board.turn == perspective else MATE
    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == perspective else -1
        file_distance = abs(chess.square_file(square) - 3.5)
        rank_distance = abs(chess.square_rank(square) - 3.5)
        center = round(14 - 2 * (file_distance + rank_distance))
        placement = center if piece.piece_type in (chess.KNIGHT, chess.BISHOP) else 0
        score += sign * (VALUES[piece.piece_type] + placement)
    return score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    perspective = board.turn
    scored: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        board.push(move)
        replies = list(board.legal_moves)
        if not replies:
            worst_reply = evaluate(board, perspective)
        else:
            worst_reply = math.inf
            for reply in replies:
                board.push(reply)
                worst_reply = min(worst_reply, evaluate(board, perspective))
                board.pop()
        board.pop()
        scored.append((int(worst_reply), move))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        raise ValueError("get_move called on a terminal position")

    # Mostly choose among plausible moves, with occasional novice-scale oversights.
    if len(scored) > 1 and RNG.random() < 0.08:
        return RNG.choice(scored)[1].uci()
    candidates = scored[: min(4, len(scored))]
    best = candidates[0][0]
    weights = [math.exp(max(-6.0, (score - best) / 85.0)) for score, _ in candidates]
    return RNG.choices([move for _, move in candidates], weights=weights, k=1)[0].uci()
