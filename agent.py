"""Original competition-safe chess agent using selective alpha-beta search."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Final

import chess
import numpy as np
from numba import njit
from numpy.typing import NDArray

INFINITY: Final = 32_000
MATE: Final = 31_000
MATE_BOUND: Final = 30_000
MAX_PLY: Final = 128
# A hard guard against pathological capture trees. Proper SEE-based pruning is the next refinement.
MAX_QPLY: Final = 8
TT_CAPACITY: Final = 180_000
EVAL_CAPACITY: Final = 60_000
PAWN_CACHE_CAPACITY: Final = 16_000
TIME_CHECK_MASK: Final = 127
MIN_PONDER_REMAINING_MS: Final = 750
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
MG_VALUE_ARRAY: Final = np.asarray(MG_VALUE, dtype=np.int32)
EG_VALUE_ARRAY: Final = np.asarray(EG_VALUE, dtype=np.int32)
PHASE_WEIGHT_ARRAY: Final = np.asarray(PHASE_WEIGHT, dtype=np.int32)
MG_TABLE_ARRAY: Final = np.asarray(MG_TABLE[1:], dtype=np.int32)
EG_TABLE_ARRAY: Final = np.asarray(EG_TABLE[1:], dtype=np.int32)
ADJACENT_FILES: Final = tuple(
    (chess.BB_FILES[file_index - 1] if file_index else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)
PASSED_MASKS: Final = tuple(
    tuple(
        (
            chess.BB_FILES[chess.square_file(square)]
            | ADJACENT_FILES[chess.square_file(square)]
        )
        & sum(
            (
                chess.BB_RANKS[rank]
                for rank in (
                    range(chess.square_rank(square) + 1, 8)
                    if color == chess.WHITE
                    else range(chess.square_rank(square))
                )
            ),
            0,
        )
        for square in chess.SQUARES
    )
    # chess.Color is bool, so tuple indices must be BLACK (0), WHITE (1).
    for color in (chess.BLACK, chess.WHITE)
)
PASSED_BONUS: Final = (0, 8, 14, 24, 40, 65, 100, 0)


@njit(cache=False)
def _compiled_piece_score(bitboards: NDArray[np.uint64]) -> tuple[int, int, int]:
    mg = 0
    eg = 0
    phase = 0
    for color_index in range(2):
        sign = -1 if color_index == 0 else 1
        for piece_index in range(6):
            pieces = bitboards[color_index * 6 + piece_index]
            piece_type = piece_index + 1
            for square in range(64):
                if pieces & (np.uint64(1) << np.uint64(square)):
                    index = square if color_index else square ^ 56
                    mg += sign * (
                        MG_VALUE_ARRAY[piece_type] + MG_TABLE_ARRAY[piece_index, index]
                    )
                    eg += sign * (
                        EG_VALUE_ARRAY[piece_type] + EG_TABLE_ARRAY[piece_index, index]
                    )
                    phase += PHASE_WEIGHT_ARRAY[piece_type]
    return mg, eg, phase


def _encode_bitboards(board: chess.Board, destination: NDArray[np.uint64]) -> None:
    for color_index, color in enumerate((chess.BLACK, chess.WHITE)):
        for piece_index, piece_type in enumerate(range(chess.PAWN, chess.KING + 1)):
            destination[color_index * 6 + piece_index] = board.pieces_mask(piece_type, color)


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
    reverse_futility_prunes: int = 0
    lmr_reductions: int = 0
    lmr_researches: int = 0
    see_prunes: int = 0
    delta_prunes: int = 0
    eval_calls: int = 0
    eval_hits: int = 0
    pawn_calls: int = 0
    pawn_hits: int = 0


def _tt_key(board: chess.Board) -> object:
    """Use the pinned python-chess repetition key without accepting hash collisions."""
    return board._transposition_key()


def static_exchange(board: chess.Board, move: chess.Move) -> int:
    """Estimate the material result of all captures on ``move.to_square``.

    This swap-off calculation is used for ordering and conservative quiescence pruning. It follows
    x-ray attacks as occupancy changes and deliberately does not treat its result as a legal-search
    score.
    """
    if not board.is_capture(move):
        return (MG_VALUE[move.promotion] - MG_VALUE[chess.PAWN]) if move.promotion else 0

    target = move.to_square
    captured = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(target)
    if captured is None:
        return 0
    promotion_gain = MG_VALUE[move.promotion] - MG_VALUE[chess.PAWN] if move.promotion else 0
    gains = [MG_VALUE[captured] + promotion_gain]
    occupied = board.occupied ^ chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        captured_square = target - 8 if board.turn == chess.WHITE else target + 8
        occupied ^= chess.BB_SQUARES[captured_square]

    side = not board.turn
    occupant_value = MG_VALUE[move.promotion or board.piece_type_at(move.from_square) or chess.PAWN]
    while len(gains) < 16:
        attackers = board.attackers_mask(side, target, occupied) & occupied
        attacker_square: chess.Square | None = None
        attacker_type = 0
        for piece_type in range(chess.PAWN, chess.KING + 1):
            candidates = attackers & board.pieces_mask(piece_type, side)
            if candidates:
                attacker_square = chess.scan_forward(candidates).__next__()
                attacker_type = piece_type
                break
        if attacker_square is None:
            break
        gains.append(occupant_value - gains[-1])
        if max(-gains[-2], gains[-1]) < 0:
            break
        occupied ^= chess.BB_SQUARES[attacker_square]
        occupant_value = MG_VALUE[attacker_type]
        side = not side

    for index in range(len(gains) - 2, -1, -1):
        gains[index] = -max(-gains[index], gains[index + 1])
    return gains[0]


def _pawn_structure(white_pawns: int, black_pawns: int) -> int:
    """Return pawn-structure terms from White's perspective."""
    score = 0
    for color, sign, pawns, enemies in (
        (chess.WHITE, 1, white_pawns, black_pawns),
        (chess.BLACK, -1, black_pawns, white_pawns),
    ):
        for file_index, file_mask in enumerate(chess.BB_FILES):
            count = (pawns & file_mask).bit_count()
            if count > 1:
                score -= sign * 11 * (count - 1)
            if count and not pawns & ADJACENT_FILES[file_index]:
                score -= sign * 9 * count
        for square in chess.scan_forward(pawns):
            if not enemies & PASSED_MASKS[color][square]:
                rank_index = chess.square_rank(square)
                advance = rank_index if color else 7 - rank_index
                score += sign * PASSED_BONUS[advance]
    return score


def evaluate(
    board: chess.Board,
    pawn_score: int | None = None,
    bitboards: NDArray[np.uint64] | None = None,
) -> int:
    """Tapered material, placement, pawn, bishop-pair, and rook-file evaluation."""
    buffer = np.empty(12, dtype=np.uint64) if bitboards is None else bitboards
    _encode_bitboards(board, buffer)
    mg, eg, phase = _compiled_piece_score(buffer)

    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    score += _pawn_structure(white_pawns, black_pawns) if pawn_score is None else pawn_score
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 28
        pawns = white_pawns if color else black_pawns
        enemies = black_pawns if color else white_pawns
        for square in board.pieces(chess.ROOK, color):
            file_mask = chess.BB_FILES[chess.square_file(square)]
            if not pawns & file_mask:
                score += sign * (17 if not enemies & file_mask else 9)
    return score if board.turn == chess.WHITE else -score


class Engine:
    """State that persists for one game and is naturally reset with the process."""

    def __init__(self) -> None:
        self.tt: dict[object, TTEntry] = {}
        self.eval_cache: dict[object, int] = {}
        self.pawn_cache: dict[tuple[int, int], int] = {}
        self.eval_bitboards = np.empty(12, dtype=np.uint64)
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.nodes = 0
        self.stats = SearchStats()
        self.deadline = 0.0
        self.age = 0
        self.cancelled = False

    def _check_time(self) -> None:
        if self.nodes & TIME_CHECK_MASK == 0 and (
            self.cancelled or time.perf_counter() >= self.deadline
        ):
            raise SearchTimeout

    def _evaluate(self, board: chess.Board) -> int:
        self.stats.eval_calls += 1
        key = _tt_key(board)
        cached = self.eval_cache.get(key)
        if cached is not None:
            self.stats.eval_hits += 1
            return cached
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
        pawn_key = (white_pawns, black_pawns)
        self.stats.pawn_calls += 1
        pawn_score = self.pawn_cache.get(pawn_key)
        if pawn_score is None:
            pawn_score = _pawn_structure(white_pawns, black_pawns)
            if len(self.pawn_cache) >= PAWN_CACHE_CAPACITY:
                self.pawn_cache.clear()
            self.pawn_cache[pawn_key] = pawn_score
        else:
            self.stats.pawn_hits += 1
        score = evaluate(board, pawn_score, self.eval_bitboards)
        if len(self.eval_cache) >= EVAL_CAPACITY:
            self.eval_cache.clear()
        self.eval_cache[key] = score
        return score

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
        if board.halfmove_clock >= 100:
            return True
        sufficient_material = board.pawns | board.rooks | board.queens
        if not sufficient_material and board.is_insufficient_material():
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
            return self._evaluate(board)
        stand_pat = -INFINITY
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            stand_pat = self._evaluate(board)
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
            if not in_check and not move.promotion:
                captured_value = self._captured_value(board, move)
                loses_to_delta = stand_pat + captured_value + 180 < alpha
                loses_to_see = (
                    not loses_to_delta and qply >= 2 and static_exchange(board, move) < -120
                )
                if (loses_to_delta or loses_to_see) and not board.gives_check(move):
                    if loses_to_delta:
                        self.stats.delta_prunes += 1
                    else:
                        self.stats.see_prunes += 1
                    continue
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
            return self._evaluate(board)

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

        static_eval: int | None = None
        has_non_pawn_material = self._has_non_pawn_material(board, board.turn)
        if not in_check and has_non_pawn_material:
            static_eval = self._evaluate(board)
            # At shallow, non-PV nodes, a position comfortably above beta does not need
            # move-by-move proof. Keep mate windows and pawn-only zugzwangs out of this rule.
            if (
                depth <= 3
                and beta - alpha == 1
                and abs(beta) < MATE_BOUND
                and static_eval - 90 * depth >= beta
            ):
                self.stats.reverse_futility_prunes += 1
                return static_eval

        if (
            allow_null
            and depth >= 3
            and not in_check
            and has_non_pawn_material
            and static_eval is not None
            and static_eval >= beta
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
            lmr_candidate = depth >= 3 and index >= 4 and quiet and not in_check
            gives_check = board.gives_check(move) if lmr_candidate else False
            board.push(move)
            try:
                if index == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    reduction = 0
                    if lmr_candidate and not gives_check:
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
        increment_share = min(0.35, usable * 0.12)
        optimum = min(usable, max(0.004, usable / 22.0 + increment_share))
        return optimum, min(usable, max(optimum, optimum * 1.9))

    def choose(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        self.cancelled = False
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
_PONDER_THREAD: threading.Thread | None = None


def _stop_pondering() -> None:
    global _PONDER_THREAD
    if _PONDER_THREAD is None:
        return
    _ENGINE.cancelled = True
    _PONDER_THREAD.join()
    _PONDER_THREAD = None


def _ponder(board: chess.Board) -> None:
    """Search the opponent-to-move position and retain the resulting TT entries."""
    try:
        _ENGINE.choose(board, 3_600_000)
    except Exception:
        # Pondering is optional acceleration and must never endanger a legal reply.
        return


def get_move(fen: str, time_left_ms: int) -> str:
    """Return one legal UCI move under the competition contract."""
    global _PONDER_THREAD
    _stop_pondering()
    board = chess.Board(fen)
    move = _ENGINE.choose(board, time_left_ms)
    board.push(move)
    estimated_remaining_ms = time_left_ms - round(_ENGINE.stats.elapsed_s * 1000)
    if not board.is_game_over() and estimated_remaining_ms >= MIN_PONDER_REMAINING_MS:
        _PONDER_THREAD = threading.Thread(target=_ponder, args=(board,), daemon=True)
        _PONDER_THREAD.start()
    return move.uci()


# Compile outside the clock during the platform's import allowance.
_warmup_bitboards = np.empty(12, dtype=np.uint64)
_encode_bitboards(chess.Board(), _warmup_bitboards)
_compiled_piece_score(_warmup_bitboards)
