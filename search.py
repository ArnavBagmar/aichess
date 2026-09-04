"""Alpha-beta search over the NNUE evaluation.

The searcher drives nnue_engine.Engine through push/pop so the incremental
accumulators keep behaving exactly as tests/test_parity.py pins them. Scores stay in
the engine's native 1/32-centipawn integer units end to end.
"""

import time
from collections.abc import Hashable
from typing import Final

import chess

from nnue_engine import Engine

# Score conventions. A mate at distance `ply` scores MATE - ply, so shorter mates win
# and mate scores stay comparable across depths.
MATE: Final = 1_000_000

MAX_DEPTH: Final = 64
MAX_PLY_LIMIT: Final = 128  # well inside nnue_engine.MAX_PLY (256)

# Scores at or beyond this magnitude are mates, and their distance is meaningful.
MATE_THRESHOLD: Final = MATE - MAX_PLY_LIMIT

TT_MAX_ENTRIES: Final = 200_000
# At ~15-20 knps this is a 15-25 ms blind spot between clock checks; 2048 was up to
# 150 ms on a loaded machine, enough to flag at a short increment.
CLOCK_CHECK_NODES: Final = 256

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

# Null-move pruning: if passing the turn still fails high, the position is good enough
# that searching it properly is wasted work.
NULL_MIN_DEPTH: Final = 3
NULL_REDUCTION: Final = 2

# Late move reductions. Moves the ordering put late are searched shallower first, and
# only re-searched at full depth if they beat alpha after all.
LMR_MIN_DEPTH: Final = 3
LMR_MIN_MOVES: Final = 4
LMR_LATE_MOVES: Final = 8
LMR_DEEP: Final = 6

# Transposition bound kinds.
EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

TTEntry = tuple[int, int, int, chess.Move | None, int]  # depth, score, bound, move, generation


def to_tt_score(score: int, ply: int) -> int:
    """Convert a root-relative mate score to a node-relative one for storage.

    A mate scores MATE - (plies from the root). Two paths reaching the same position
    at different plies must agree on the entry, so the table holds MATE - (plies from
    this node) instead, and the probe converts back.
    """
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def from_tt_score(score: int, ply: int) -> int:
    """Inverse of to_tt_score, for a probe at `ply`."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


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
        self.generation = 0  # one per pick, so replacement can prefer this search's work
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

    def pick(self, fen: str, time_left_ms: int) -> chess.Move:
        """Best move for `fen`, within the budget implied by `time_left_ms`."""
        self.engine.set_position(fen)
        board = self.engine.board
        moves = self.ordered_moves(board, 0, None)
        if not moves:
            raise ValueError(f"no legal moves in {fen!r}")

        self.note_root_position(board)
        self._deadline = time.monotonic() + budget_ms(time_left_ms) / 1000.0
        self.nodes = 0
        self.generation += 1
        self._path.clear()

        best = moves[0]
        for depth in range(1, MAX_DEPTH):
            try:
                score, move = self._search_root(depth, best)
            except SearchAborted:
                break  # discard this depth entirely; it has a biased best move
            if move is not None:
                best = move
            if abs(score) >= MATE_THRESHOLD:
                break  # a forced mate is as good as it gets
        return best

    def _search_root(
        self,
        depth: int,
        previous_best: chess.Move | None,
        alpha: int = -2 * MATE,
        beta: int = 2 * MATE,
    ) -> tuple[int, chess.Move | None]:
        board = self.engine.board
        best_score = -2 * MATE
        best_move: chess.Move | None = None
        for index, move in enumerate(self.ordered_moves(board, 0, previous_best)):
            self.engine.push(move)
            try:
                score = self._pvs_child(index, depth - 1, 1, alpha, beta, reduction=0)
            finally:
                self.engine.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_move

    def _pvs_child(
        self, index: int, depth: int, ply: int, alpha: int, beta: int, reduction: int
    ) -> int:
        """Score the child already pushed, with the principal-variation scheme.

        The first move gets the full window. Every later move is first asked only
        "are you better than alpha?" with a null window, which is much cheaper; only a
        yes earns a full search. A reduced move that says yes is first confirmed at full
        depth with the null window, then, if still yes, with the full window.
        """
        if index == 0:
            return -self._negamax(depth, ply, -beta, -alpha)
        score = -self._negamax(depth - reduction, ply, -alpha - 1, -alpha)
        if reduction and score > alpha:
            score = -self._negamax(depth, ply, -alpha - 1, -alpha)
        if alpha < score < beta:
            score = -self._negamax(depth, ply, -beta, -alpha)
        return score

    def _check_clock(self) -> None:
        self.nodes += 1
        if self.nodes % CLOCK_CHECK_NODES == 0 and time.monotonic() > self._deadline:
            raise SearchAborted

    def _negamax(
        self, depth: int, ply: int, alpha: int, beta: int, allow_null: bool = True
    ) -> int:
        self._check_clock()
        board = self.engine.board
        if self.is_draw(board, ply):
            return 0

        key = board._transposition_key()
        tt_move: chess.Move | None = None
        entry = self.table.get(key)
        if entry is not None:
            stored_depth, raw_score, stored_bound, tt_move, _ = entry
            if stored_depth >= depth:
                stored_score = from_tt_score(raw_score, ply)
                if stored_bound == EXACT:
                    return stored_score
                if stored_bound == LOWER and stored_score >= beta:
                    return stored_score
                if stored_bound == UPPER and stored_score <= alpha:
                    return stored_score

        if depth <= 0 or ply >= MAX_PLY_LIMIT:
            return self._quiescence(ply, alpha, beta)

        moves = self.ordered_moves(board, ply, tt_move)
        in_check = board.is_check()
        if not moves:
            return -MATE + ply if in_check else 0

        # Null move. Moves are generated first so a stalemate cannot be mistaken for a
        # fail-high: passing is only meaningful when there was something to pass up.
        if (
            allow_null
            and depth >= NULL_MIN_DEPTH
            and not in_check
            and beta < MATE_THRESHOLD
            and self._has_major_material(board)
        ):
            self.engine.push_null()
            try:
                passed = -self._negamax(
                    depth - 1 - NULL_REDUCTION, ply + 1, -beta, -beta + 1, allow_null=False
                )
            finally:
                self.engine.pop()
            if passed >= beta:
                # Deliberately beta, not `passed`: a score borrowed from a position that
                # skipped a turn is not trustworthy enough to store as the real value.
                return beta

        original_alpha = alpha
        best_score = -2 * MATE
        best_move: chess.Move | None = None
        self._path.append(key)
        try:
            for index, move in enumerate(moves):
                tactical = board.is_capture(move) or move.promotion is not None
                self.engine.push(move)
                try:
                    reduction = 0
                    if (
                        depth >= LMR_MIN_DEPTH
                        and index >= LMR_MIN_MOVES
                        and not tactical
                        and not in_check
                        and not self.engine.board.is_check()
                    ):
                        reduction = self._late_move_reduction(depth, index)
                    score = self._pvs_child(index, depth - 1, ply + 1, alpha, beta, reduction)
                finally:
                    self.engine.pop()
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, score)
                if alpha >= beta:
                    self._remember_cutoff(board, move, ply, depth)
                    break
        finally:
            self._path.pop()

        if best_score <= original_alpha:
            bound = UPPER
        elif best_score >= beta:
            bound = LOWER
        else:
            bound = EXACT
        self._store(key, depth, best_score, bound, best_move, ply)
        return best_score

    @staticmethod
    def _has_major_material(board: chess.Board) -> bool:
        """Whether the side to move has a piece other than pawns and the king.

        Null-move pruning assumes passing is never better than moving, which is exactly
        false in zugzwang — and zugzwang is overwhelmingly a king-and-pawn affair.
        """
        pieces = board.knights | board.bishops | board.rooks | board.queens
        return bool(pieces & board.occupied_co[board.turn])

    @staticmethod
    def _late_move_reduction(depth: int, index: int) -> int:
        """How much to shave off a late quiet move. Deeper and later means bolder."""
        if index >= LMR_LATE_MOVES and depth >= LMR_DEEP:
            return 2
        return 1

    def _store(
        self,
        key: Hashable,
        depth: int,
        score: int,
        bound: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        existing = self.table.get(key)
        if existing is not None:
            if existing[4] == self.generation and existing[0] > depth:
                return  # a deeper result from this very search outranks a shallower one
        elif len(self.table) >= TT_MAX_ENTRIES:
            # Crude, but it bounds memory well inside 2 GB and costs nothing on the
            # hot path. Refilling is cheap next to running out of memory.
            self.table.clear()
        self.table[key] = (depth, to_tt_score(score, ply), bound, move, self.generation)

    def _quiescence(self, ply: int, alpha: int, beta: int) -> int:
        """Search captures to a quiet position, so the evaluation is not mid-exchange."""
        self._check_clock()
        board = self.engine.board
        in_check = board.is_check()

        if in_check:
            # No stand-pat while in check: the position may be a forced mate.
            moves = self.ordered_moves(board, ply, None)
            if not moves:
                return -MATE + ply
            best_score = -2 * MATE
        else:
            best_score = self.engine.evaluate()
            if best_score >= beta:
                return best_score
            alpha = max(alpha, best_score)
            moves = self.ordered_captures(board)

        if ply >= MAX_PLY_LIMIT:
            return self.engine.evaluate()

        for move in moves:
            self.engine.push(move)
            try:
                score = -self._quiescence(ply + 1, -beta, -alpha)
            finally:
                self.engine.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def warm_up(self) -> None:
        """Compile the search's own code paths inside the import budget.

        numba compiles per signature, and the platform gives 60 s before the clock
        starts; paying that here is the whole point. Leaves no state behind.
        """
        self.pick(chess.STARTING_FEN, 200)
        self.table.clear()
        self.history.clear()
        self.killers = [[] for _ in range(MAX_PLY_LIMIT + 1)]
        self.game_history.clear()
        self._last_fullmove = 0
