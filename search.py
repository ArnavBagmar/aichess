"""Alpha-beta search over the NNUE evaluation.

The searcher drives nnue_engine.Engine through push/pop so the incremental
accumulators keep behaving exactly as tests/test_parity.py pins them. Scores stay in
the engine's native 1/32-centipawn integer units end to end.
"""

from collections.abc import Hashable
from typing import Final

import chess

from nnue_engine import Engine

# Score conventions. A mate at distance `ply` scores MATE - ply, so shorter mates win
# and mate scores stay comparable across depths.
MATE: Final = 1_000_000

MAX_DEPTH: Final = 64
MAX_PLY_LIMIT: Final = 128  # well inside nnue_engine.MAX_PLY (256)

TT_MAX_ENTRIES: Final = 200_000
CLOCK_CHECK_NODES: Final = 2048

# Time control. Every term but INCREMENT_MS comes from the clock we were handed.
MOVES_REMAINING: Final = 30  # assumed horizon; self-correcting as the clock changes
INCREMENT_MS: Final = 500  # published time control: 120 s + 0.5 s per move
SAFETY_MS: Final = 50  # margin; the referee measures wall time and does not forgive
MAX_FRACTION: Final = 0.4  # never spend more than this much of what is left
MIN_BUDGET_MS: Final = 10  # always attempt something


class SearchAborted(Exception):
    """The move budget is spent; unwinds to the iterative-deepening loop."""


def budget_ms(time_left_ms: int) -> float:
    """Milliseconds to spend on this move, derived from the clock we were handed.

    Only a fraction of INCREMENT_MS is claimed: it is the one number taken from the
    published rules rather than the input, so if the platform ever changed the
    increment the result is a conservative budget rather than a flag.
    """
    budget = time_left_ms / MOVES_REMAINING + 0.6 * INCREMENT_MS
    budget = min(budget, MAX_FRACTION * time_left_ms)
    return max(budget - SAFETY_MS, MIN_BUDGET_MS)


# MVV-LVA victim/attacker values, in the usual pawn-to-king order.
PIECE_VALUE: Final = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 20,
}

TT_MOVE_BONUS: Final = 1_000_000
CAPTURE_BONUS: Final = 500_000
KILLER_BONUS: Final = 400_000
KILLERS_PER_PLY: Final = 2

# Transposition bound kinds.
EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

TTEntry = tuple[int, int, int, chess.Move | None]  # depth, score, bound, best move


class Searcher:
    """Alpha-beta search state for one game.

    The process lives for a single game, so the table and history accumulate freely
    across our moves and can never leak into another game.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.table: dict[Hashable, TTEntry] = {}
        self.killers: list[list[chess.Move]] = [[] for _ in range(MAX_PLY_LIMIT + 1)]
        self.history: dict[tuple[int, int], int] = {}
        self.game_history: list[Hashable] = []
        self.nodes = 0
        self._deadline = 0.0
        self._path: list[Hashable] = []
        self._last_fullmove = 0

    def note_root_position(self, board: chess.Board) -> None:
        """Record a position we were asked about, for repetition detection.

        We only ever observe positions where it is our turn — every other ply — which
        is enough: a position repeating at our turn is a genuine repetition.
        """
        if board.fullmove_number < self._last_fullmove:
            # Cannot follow the previous root. Never expected within one game; this
            # exists so a surprise cannot become a bogus draw claim.
            self.game_history.clear()
        self._last_fullmove = board.fullmove_number
        self.game_history.append(board._transposition_key())

    def is_draw(self, board: chess.Board, ply: int) -> bool:
        """Whether this node should score as a draw. Never true at the root."""
        if ply == 0:
            return False
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return True
        key = board._transposition_key()
        return key in self._path or key in self.game_history

    def _capture_score(self, board: chess.Board, move: chess.Move) -> int:
        """MVV-LVA: most valuable victim first, cheapest attacker as the tiebreak."""
        victim = board.piece_type_at(move.to_square)
        if victim is None:  # en passant leaves the target square empty
            victim = chess.PAWN
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        score = 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
        if move.promotion is not None:
            score += 10 * PIECE_VALUE[move.promotion]
        return score

    def _move_score(
        self, board: chess.Board, move: chess.Move, ply: int, tt_move: chess.Move | None
    ) -> int:
        if tt_move is not None and move == tt_move:
            return TT_MOVE_BONUS
        if board.is_capture(move) or move.promotion is not None:
            return CAPTURE_BONUS + self._capture_score(board, move)
        if move in self.killers[ply]:
            return KILLER_BONUS
        return self.history.get((move.from_square, move.to_square), 0)

    def ordered_moves(
        self, board: chess.Board, ply: int, tt_move: chess.Move | None
    ) -> list[chess.Move]:
        """Every legal move, best guess first."""
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._move_score(board, move, ply, tt_move), reverse=True)
        return moves

    def ordered_captures(self, board: chess.Board) -> list[chess.Move]:
        """Captures and queen promotions only, for quiescence."""
        moves = list(board.generate_legal_captures())
        moves += [
            move
            for move in board.generate_legal_moves()
            if move.promotion == chess.QUEEN and not board.is_capture(move)
        ]
        moves.sort(key=lambda move: self._capture_score(board, move), reverse=True)
        return moves

    def _remember_cutoff(self, board: chess.Board, move: chess.Move, ply: int, depth: int) -> None:
        """Record a quiet move that caused a beta cutoff, for future ordering."""
        if board.is_capture(move) or move.promotion is not None:
            return
        killers = self.killers[ply]
        if move not in killers:
            killers.insert(0, move)
            del killers[KILLERS_PER_PLY:]
        key = (move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth
