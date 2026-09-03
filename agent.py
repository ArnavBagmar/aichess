"""Original competition-safe chess agent using selective alpha-beta search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import chess

INFINITY: Final = 32_000
MATE: Final = 31_000
MATE_BOUND: Final = 30_000
MAX_PLY: Final = 128
# A hard guard against pathological capture trees. Proper SEE-based pruning is the next refinement.
MAX_QPLY: Final = 8
TT_CAPACITY: Final = 180_000
TIME_CHECK_MASK: Final = 127
EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

MG_VALUE: Final = (0, 100, 320, 335, 500, 930, 0)
EG_VALUE: Final = (0, 120, 305, 330, 525, 940, 0)
PHASE_WEIGHT: Final = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE: Final = 24


def _table(center: int, edge: int, advance: int = 0) -> tuple[int, ...]:
    values: list[int] = []
    for rank in range(8):
        for file_index in range(8):
            distance = abs(file_index - 3.5) + abs(rank - 3.5)
            values.append(round(center + (edge - center) * distance / 7 + advance * rank))
    return tuple(values)


PAWN_MG: Final = tuple(
    0 if rank in (0, 7) else (rank * 7 + (12 if file_index in (3, 4) else 0))
    for rank in range(8)
    for file_index in range(8)
)
PAWN_EG: Final = tuple(
    0 if rank in (0, 7) else rank * rank * 3 for rank in range(8) for _ in range(8)
)
KNIGHT_MG: Final = _table(24, -48)
KNIGHT_EG: Final = _table(18, -35)
BISHOP_MG: Final = _table(14, -18)
BISHOP_EG: Final = _table(12, -14)
ROOK_MG: Final = tuple(8 if rank == 6 else 0 for rank in range(8) for _ in range(8))
ROOK_EG: Final = _table(6, 0)
QUEEN_MG: Final = _table(6, -12)
QUEEN_EG: Final = _table(8, -10)
KING_MG: Final = tuple(
    (28 if rank == 0 and file_index in (1, 2, 6) else -8 * rank)
    for rank in range(8)
    for file_index in range(8)
)
KING_EG: Final = _table(28, -42)
MG_TABLE: Final = ((), PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG)
EG_TABLE: Final = ((), PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG)


class SearchTimeout(Exception):
    """Stop the current iteration while preserving the last completed move."""


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: int
    move: chess.Move | None
    age: int


@dataclass(slots=True)
class SearchStats:
    """Low-overhead counters for tests and offline profiling."""

    nodes: int = 0
    qnodes: int = 0
    completed_depth: int = 0
    score: int = 0
    elapsed_s: float = 0.0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    beta_cutoffs: int = 0
    first_move_cutoffs: int = 0
    null_tries: int = 0
    null_cutoffs: int = 0
    lmr_reductions: int = 0
    lmr_researches: int = 0


def _tt_key(board: chess.Board) -> object:
    """Use the pinned python-chess repetition key without accepting hash collisions."""
    return board._transposition_key()


def evaluate(board: chess.Board) -> int:
    """Tapered material, placement, pawn, bishop-pair, and rook-file evaluation."""
    mg = eg = phase = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        piece_type = piece.piece_type
        mg += sign * (MG_VALUE[piece_type] + MG_TABLE[piece_type][index])
        eg += sign * (EG_VALUE[piece_type] + EG_TABLE[piece_type][index])
        phase += PHASE_WEIGHT[piece_type]

    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 28
        pawns = board.pieces(chess.PAWN, color)
        enemies = board.pieces(chess.PAWN, not color)
        for file_index in range(8):
            file_mask = chess.BB_FILES[file_index]
            count = len(pawns & file_mask)
            if count > 1:
                score -= sign * 11 * (count - 1)
            adjacent_files = 0
            if file_index:
                adjacent_files |= chess.BB_FILES[file_index - 1]
            if file_index < 7:
                adjacent_files |= chess.BB_FILES[file_index + 1]
            if count and not pawns & adjacent_files:
                score -= sign * 9 * count
        for square in pawns:
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            files = chess.BB_FILES[file_index]
            if file_index:
                files |= chess.BB_FILES[file_index - 1]
            if file_index < 7:
                files |= chess.BB_FILES[file_index + 1]
            forward = 0
            ranks = range(rank_index + 1, 8) if color else range(rank_index)
            for rank in ranks:
                forward |= chess.BB_RANKS[rank]
            if not enemies & files & forward:
                advance = rank_index if color else 7 - rank_index
                score += sign * (0, 8, 14, 24, 40, 65, 100, 0)[advance]
        for square in board.pieces(chess.ROOK, color):
            file_mask = chess.BB_FILES[chess.square_file(square)]
            if not pawns & file_mask:
                score += sign * (17 if not enemies & file_mask else 9)
    return score if board.turn == chess.WHITE else -score


class Engine:
    """State that persists for one game and is naturally reset with the process."""

    def __init__(self) -> None:
        self.tt: dict[object, TTEntry] = {}
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.nodes = 0
        self.stats = SearchStats()
        self.deadline = 0.0
        self.age = 0

    def _check_time(self) -> None:
        if self.nodes & TIME_CHECK_MASK == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
        return bool(
            board.pieces(chess.KNIGHT, color)
            | board.pieces(chess.BISHOP, color)
            | board.pieces(chess.ROOK, color)
            | board.pieces(chess.QUEEN, color)
        )

    @staticmethod
    def _captured_value(board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return MG_VALUE[chess.PAWN]
        victim = board.piece_type_at(move.to_square)
        return MG_VALUE[victim] if victim is not None else 0

    def _move_score(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return 2_000_000
        if move.promotion:
            return 1_500_000 + MG_VALUE[move.promotion]
        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            return 1_000_000 + 16 * self._captured_value(board, move) - MG_VALUE[attacker]
        if ply < MAX_PLY and move == self.killers[ply][0]:
            return 900_000
        if ply < MAX_PLY and move == self.killers[ply][1]:
            return 800_000
        return self.history.get((board.turn, move.from_square, move.to_square), 0)

    def _ordered_moves(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        moves.sort(key=lambda move: self._move_score(board, move, tt_move, ply), reverse=True)
        return moves

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        return score + ply if score > MATE_BOUND else score - ply if score < -MATE_BOUND else score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        return score - ply if score > MATE_BOUND else score + ply if score < -MATE_BOUND else score

    def _store(self, key: object, entry: TTEntry) -> None:
        old = self.tt.get(key)
        if old is None and len(self.tt) >= TT_CAPACITY:
            self.tt.clear()
        if old is None or entry.depth >= old.depth or entry.age > old.age:
            self.tt[key] = entry

    @staticmethod
    def _is_draw(board: chess.Board) -> bool:
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return True
        return len(board.move_stack) >= 8 and board.is_repetition(3)

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int) -> int:
        self.nodes += 1
        self.stats.qnodes += 1
        self._check_time()
        if self._is_draw(board):
            return 0
        in_check = board.is_check()
        if ply >= MAX_PLY or qply >= MAX_QPLY:
            return evaluate(board)
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = list(board.generate_legal_captures())
            promotion_ranks = chess.BB_RANK_1 | chess.BB_RANK_8
            moves.extend(
                move
                for move in board.generate_legal_moves(
                    from_mask=board.pawns, to_mask=promotion_ranks
                )
                if not board.is_capture(move)
            )
            if not moves and board.is_stalemate():
                return 0
        for move in self._ordered_moves(board, moves, None, ply):
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool
    ) -> int:
        self.nodes += 1
        self._check_time()
        if self._is_draw(board):
            return 0
        if ply >= MAX_PLY:
            return evaluate(board)

        alpha_original = alpha
        key = _tt_key(board)
        self.stats.tt_probes += 1
        entry = self.tt.get(key)
        tt_move = entry.move if entry else None
        if entry is not None:
            self.stats.tt_hits += 1
        if entry is not None and entry.depth >= depth:
            tt_score = self._score_from_tt(entry.score, ply)
            if entry.flag == EXACT:
                return tt_score
            if entry.flag == LOWER:
                alpha = max(alpha, tt_score)
            else:
                beta = min(beta, tt_score)
            if alpha >= beta:
                self.stats.tt_cutoffs += 1
                return tt_score

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)
        in_check = board.is_check()
        if in_check and depth <= 6:
            depth += 1

        if (
            allow_null
            and depth >= 3
            and not in_check
            and self._has_non_pawn_material(board, board.turn)
            and evaluate(board) >= beta
        ):
            self.stats.null_tries += 1
            reduction = 2 + depth // 4
            board.push(chess.Move.null())
            try:
                score = -self._negamax(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
                )
            finally:
                board.pop()
            if score >= beta:
                self.stats.null_cutoffs += 1
                return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        moves = self._ordered_moves(board, moves, tt_move, ply)
        best_score = -INFINITY
        best_move: chess.Move | None = None
        for index, move in enumerate(moves):
            quiet = not board.is_capture(move) and move.promotion is None
            gives_check = board.gives_check(move)
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    reduction = 0
                    if depth >= 3 and index >= 4 and quiet and not in_check and not gives_check:
                        reduction = 1 + int(depth >= 6 and index >= 8)
                        self.stats.lmr_reductions += 1
                    score = -self._negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha and reduction:
                        self.stats.lmr_researches += 1
                        score = -self._negamax(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1, True
                        )
                    if score > alpha and score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                self.stats.beta_cutoffs += 1
                if index == 0:
                    self.stats.first_move_cutoffs += 1
                if quiet:
                    if ply < MAX_PLY and move != self.killers[ply][0]:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = move
                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = min(
                        1_000_000, self.history.get(history_key, 0) + depth * depth
                    )
                break

        flag = UPPER if best_score <= alpha_original else LOWER if best_score >= beta else EXACT
        self._store(
            key, TTEntry(depth, self._score_to_tt(best_score, ply), flag, best_move, self.age)
        )
        return best_score

    def _root(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> tuple[int, chess.Move]:
        key = _tt_key(board)
        entry = self.tt.get(key)
        moves = self._ordered_moves(
            board, list(board.legal_moves), entry.move if entry else None, 0
        )
        if not moves:
            raise ValueError("get_move called on a terminal position")
        best_move, best_score, alpha_original = moves[0], -INFINITY, alpha
        for index, move in enumerate(moves):
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
                else:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                    if score > alpha and score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
            finally:
                board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        flag = UPPER if best_score <= alpha_original else LOWER if best_score >= beta else EXACT
        self._store(key, TTEntry(depth, best_score, flag, best_move, self.age))
        return best_score, best_move

    @staticmethod
    def _time_budget(time_left_ms: int) -> tuple[float, float]:
        remaining = max(0.0, time_left_ms / 1000.0)
        reserve = max(0.025, min(0.75, remaining * 0.025))
        usable = max(0.001, remaining - reserve)
        # The 0.5-second increment is valuable at normal clocks but must be tapered in emergencies:
        # blindly spending most of an increment when only a second remains invites flag losses.
        increment_share = min(0.35, usable * 0.08)
        optimum = min(usable, max(0.004, usable / 28.0 + increment_share))
        return optimum, min(usable, max(optimum, optimum * 1.9))

    def choose(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        call_start = time.perf_counter()
        self.nodes = 0
        self.stats = SearchStats()
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("get_move called on a terminal position")
        fallback = moves[0]
        if len(moves) == 1 or time_left_ms <= 5:
            self.stats.elapsed_s = time.perf_counter() - call_start
            return fallback

        soft_budget, hard_budget = self._time_budget(time_left_ms)
        start = time.perf_counter()
        self.deadline = start + hard_budget
        self.age += 1
        best_move, previous_score = fallback, 0
        for depth in range(1, 65):
            try:
                if depth >= 4:
                    window = 35
                    while True:
                        score, candidate = self._root(
                            board, depth, previous_score - window, previous_score + window
                        )
                        if score <= previous_score - window or score >= previous_score + window:
                            window *= 2
                            if window >= INFINITY:
                                score, candidate = self._root(board, depth, -INFINITY, INFINITY)
                                break
                        else:
                            break
                else:
                    score, candidate = self._root(board, depth, -INFINITY, INFINITY)
            except SearchTimeout:
                break
            best_move, previous_score = candidate, score
            self.stats.completed_depth = depth
            self.stats.score = score
            if abs(score) >= MATE_BOUND or time.perf_counter() - start >= soft_budget:
                break
        self.stats.nodes = self.nodes
        self.stats.elapsed_s = time.perf_counter() - start
        return best_move


_ENGINE = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return one legal UCI move under the competition contract."""
    board = chess.Board(fen)
    return _ENGINE.choose(board, time_left_ms).uci()
