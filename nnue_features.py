"""HalfKAv2_hm feature indexing over python-chess boards.

Transcribed from nnue-pytorch's model/modules/features/halfka_v2_hm.py (the 12-plane
training layout, 24,576 inputs). A feature index for one perspective is

    oriented_square + 64 * piece_index + 768 * king_bucket

where orientation mirrors horizontally when the perspective's king is on files a-d and
vertically for the black perspective, so the king always sits on files e-h from a white
point of view. Because of that mirroring, any move of the perspective's own king can
change every index at once; callers must rebuild the accumulator instead of updating it.
"""

import chess

from nnue_arch import NUM_PLANES, NUM_SQ

# Bucket for an oriented king square; -1 marks squares the orientation makes unreachable.
# fmt: off
KING_BUCKETS = (
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0,
)
# fmt: on


def orient(white_pov: bool, square: int, king_square: int) -> int:
    """Orient a square: mirror to the king's half-board, flip ranks for black."""
    horizontal = 7 if chess.square_file(king_square) < 4 else 0
    vertical = 0 if white_pov else 56
    return square ^ horizontal ^ vertical


def piece_index(white_pov: bool, piece: chess.Piece) -> int:
    """Plane number 0..11: own pieces on even planes, enemy pieces on odd ones."""
    own = piece.color == chess.WHITE if white_pov else piece.color == chess.BLACK
    return (piece.piece_type - 1) * 2 + (0 if own else 1)


def feature_index(white_pov: bool, king_square: int, square: int, piece: chess.Piece) -> int:
    """Index of one (square, piece) pair seen from one perspective."""
    oriented_king = orient(white_pov, king_square, king_square)
    bucket = KING_BUCKETS[oriented_king]
    return (
        orient(white_pov, square, king_square)
        + NUM_SQ * piece_index(white_pov, piece)
        + NUM_PLANES * bucket
    )


def active_features(board: chess.Board, white_pov: bool) -> list[int]:
    """All active feature indices for one perspective, kings included."""
    king_square = board.king(chess.WHITE if white_pov else chess.BLACK)
    if king_square is None:
        raise ValueError("board has no king for the requested perspective")
    return [
        feature_index(white_pov, king_square, square, piece)
        for square, piece in board.piece_map().items()
    ]


def move_deltas(
    board: chess.Board, move: chess.Move, white_pov: bool
) -> tuple[list[int], list[int]] | None:
    """Feature indices (added, removed) for `move` from one perspective.

    `board` must be the position before the move. Returns None when the perspective's own
    king moves (castling included): orientation or bucket changes then, and the caller
    must rebuild from `active_features` on the new board instead.
    """
    pov_color = chess.WHITE if white_pov else chess.BLACK
    king_square = board.king(pov_color)
    if king_square is None:
        raise ValueError("board has no king for the requested perspective")

    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")
    if mover.color != board.turn:
        # push() would silently recolor the piece; fail loudly instead.
        raise ValueError(f"{move.uci()} moves a {'white' if mover.color else 'black'} piece "
                         "out of turn")
    if mover.piece_type == chess.KING and mover.color == pov_color:
        return None

    added: list[int] = []
    removed: list[int] = []

    def add(square: int, piece: chess.Piece) -> None:
        added.append(feature_index(white_pov, king_square, square, piece))

    def remove(square: int, piece: chess.Piece) -> None:
        removed.append(feature_index(white_pov, king_square, square, piece))

    remove(move.from_square, mover)
    landed = chess.Piece(move.promotion, mover.color) if move.promotion else mover
    add(move.to_square, landed)

    if board.is_en_passant(move):
        captured_square = move.to_square + (-8 if mover.color == chess.WHITE else 8)
        remove(captured_square, chess.Piece(chess.PAWN, not mover.color))
    else:
        captured = board.piece_at(move.to_square)
        if captured is not None:
            remove(move.to_square, captured)

    if board.is_castling(move):
        # Standard chess only: the enemy king lands on g/c and the rook on f/d of its
        # back rank. Our own castling never reaches here (own king move returns None).
        back_rank = chess.square_rank(move.from_square) * 8
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from = back_rank + (7 if kingside else 0)
        rook_to = back_rank + (5 if kingside else 3)
        rook = chess.Piece(chess.ROOK, mover.color)
        remove(rook_from, rook)
        add(rook_to, rook)

    return added, removed
