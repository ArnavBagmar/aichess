"""The alpha-beta search as one recursive numba kernel over the bitboard stacks.

Quiescence is the depth <= 0 branch of the same function, so the only recursion is
self-recursion, which numba resolves from the base cases. Every constant and rule is
the python-chess search's at 74a7d63; the kernel exists to run them ten times faster,
and the first gate measures exactly that.

Scores are integers in 1/32 cp for the side to move. A mate at distance `ply` scores
MATE - ply; the table stores mates relative to the node and converts on probe.

ctrl layout: [nodes, abort, root best move, node limit (0 = none), root hint move,
generation, game key count, unused]. The kernel never raises: the clock sets ctrl[1]
and every level unwinds on it.
"""

import time
from collections.abc import Callable
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from numba import njit, objmode

from bitboard import (
    BISHOP,
    EMPTY,
    FLAG_CAPTURE,
    FLAG_EP,
    HALFMOVE,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    STM,
    WHITE,
    WKING,
    ZERO,
    attackers_to,
    bit,
    generate_moves,
    is_attacked,
    lsb,
    make_move,
    make_null,
    popcount,
)
from nnue_arch import SCORE_PER_CP
from nnue_bitboard import copy_ply, evaluate, update_ply

# Score conventions. A mate at distance `ply` scores MATE - ply, so shorter mates win
# and mate scores stay comparable across depths.
MATE: Final = 1_000_000
MAX_DEPTH: Final = 64
MAX_PLY_LIMIT: Final = 128  # well inside bitboard.MAX_PLY (256)
MATE_THRESHOLD: Final = MATE - MAX_PLY_LIMIT

TT_BITS: Final = 20
TT_SIZE: Final = 1 << TT_BITS
# At a few hundred knps this is a ~10 ms blind spot between clock checks.
CLOCK_CHECK_NODES: Final = 4096

# Ordering scores; only their order matters.
TT_MOVE_BONUS: Final = 1_000_000
CAPTURE_BONUS: Final = 500_000
KILLER_BONUS: Final = 400_000
LOSING_CAPTURE_BONUS: Final = 300_000  # captures SEE calls losing: after killers

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

# The net does not score in nominal centipawns (a pawn is ~185, a knight ~1660 on
# Engine.evaluate_cp); every margin below is sized to that scale, as in phase 5.
RFP_MAX_DEPTH: Final = 3
RFP_MARGIN: Final = 120 * SCORE_PER_CP
DELTA_MARGIN: Final = 400 * SCORE_PER_CP

# Transposition bound kinds.
EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

# Indexed by piece type. MVV-LVA values in pawns; delta-pruning victim values in the
# net's units, rounded up so pruning errs on the side of searching.
PIECE_VALUE: Final = np.array([1, 3, 3, 5, 9, 20], dtype=np.int64)
VICTIM_VALUE: Final = np.array([200, 1700, 1700, 1900, 3300, 0], dtype=np.int64) * SCORE_PER_CP

CTRL_NODES: Final = 0
CTRL_ABORT: Final = 1
CTRL_ROOT_MOVE: Final = 2
CTRL_NODE_LIMIT: Final = 3
CTRL_ROOT_HINT: Final = 4
CTRL_GENERATION: Final = 5
CTRL_GAME_KEYS: Final = 6
CTRL_SIZE: Final = 8


def _jit[F: Callable[..., Any]](function: F) -> F:
    """Typed facade over numba.njit so call sites keep their signatures for mypy."""
    return cast("F", njit(cache=False)(function))


@_jit
def now() -> float:
    with objmode(t="float64"):
        t = time.monotonic()
    return float(t)


@_jit
def to_tt_score(score: int, ply: int) -> int:
    """Root-relative mate score to node-relative, for storage."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


@_jit
def from_tt_score(score: int, ply: int) -> int:
    """Inverse of to_tt_score, for a probe at `ply`."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


@_jit
def pack_tt(depth: int, bound: int, generation: int, move: int, score: int) -> int:
    """One int64: score in the high 32 bits, move, depth, bound and generation below."""
    low = (move & 0x7FFFF) | ((depth & 0x7F) << 19) | ((bound & 3) << 26)
    low |= (generation & 15) << 28
    return (score << 32) | low


@_jit
def tt_move(data: int) -> int:
    return data & 0x7FFFF


@_jit
def tt_depth(data: int) -> int:
    return (data >> 19) & 0x7F


@_jit
def tt_bound(data: int) -> int:
    return (data >> 26) & 3


@_jit
def tt_generation(data: int) -> int:
    return (data >> 28) & 15


@_jit
def tt_score(data: int) -> int:
    return data >> 32


@_jit
def is_tactical(move: int) -> bool:
    return (move >> 15) & FLAG_CAPTURE != 0 or (move >> 12) & 7 != 0


@_jit
def victim_type(mailbox_row: npt.NDArray[np.int8], move: int) -> int:
    if (move >> 15) & FLAG_EP:
        return PAWN
    code = mailbox_row[(move >> 6) & 63]
    return PAWN if code == EMPTY else int(code % 6)


@_jit
def _least_valuable(pieces_row: npt.NDArray[np.uint64], attackers: np.uint64, colour: int) -> int:
    """Piece type of the cheapest attacker of `colour` within `attackers`, or -1."""
    for piece in range(6):
        if attackers & pieces_row[colour * 6 + piece] != ZERO:
            return piece
    return -1


@_jit
def see(
    pieces_row: npt.NDArray[np.uint64],
    mailbox_row: npt.NDArray[np.int8],
    stm: int,
    occ: np.uint64,
    move: int,
    gain: npt.NDArray[np.int64],
) -> int:
    """Material won by `move` after every sensible recapture, in pawn units.

    The classic swap list: each side captures with its least valuable attacker in
    turn, sliders behind the capturer join in as it leaves the line, and each side
    may stop when continuing would lose material. The full list is walked and folded
    from the end; the usual early-exit shortcut is left out because it returns the
    wrong value when a piece behind the capturer joins in.
    """
    frm = move & 63
    to = (move >> 6) & 63
    flags = move >> 15
    piece = int(mailbox_row[frm] % 6)
    gain[0] = PIECE_VALUE[victim_type(mailbox_row, move)] if flags & FLAG_CAPTURE else 0
    occ &= ~bit(frm)
    if flags & FLAG_EP:
        occ &= ~bit(to - 8 if stm == WHITE else to + 8)
    side = stm ^ 1
    depth = 1
    while True:
        # Speculative: what the next capturer nets if it takes and is taken back.
        gain[depth] = PIECE_VALUE[piece] - gain[depth - 1]
        attackers = attackers_to(pieces_row, to, occ) & occ
        piece = _least_valuable(pieces_row, attackers, side)
        if piece < 0 or depth >= 30:
            break
        candidates = attackers & pieces_row[side * 6 + piece]
        occ &= ~bit(lsb(candidates))
        side ^= 1
        depth += 1
    depth -= 1  # the last entry assumed a recapture that never came
    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int(gain[0])


@_jit
def has_major_material(pieces_row: npt.NDArray[np.uint64], stm: int) -> bool:
    """Whether the side to move has a piece other than pawns and the king.

    Null-move pruning assumes passing is never better than moving, which is exactly
    false in zugzwang, and zugzwang is overwhelmingly a king-and-pawn affair.
    """
    base = stm * 6
    majors = pieces_row[base + KNIGHT] | pieces_row[base + BISHOP]
    majors |= pieces_row[base + ROOK] | pieces_row[base + QUEEN]
    return bool(majors != ZERO)


@_jit
def insufficient_material(pieces_row: npt.NDArray[np.uint64]) -> bool:
    """No pawns, rooks or queens, and at most one minor piece on the board."""
    heavy = pieces_row[PAWN] | pieces_row[6 + PAWN] | pieces_row[ROOK] | pieces_row[6 + ROOK]
    heavy |= pieces_row[QUEEN] | pieces_row[6 + QUEEN]
    if heavy != ZERO:
        return False
    minors = pieces_row[KNIGHT] | pieces_row[BISHOP] | pieces_row[6 + KNIGHT]
    minors |= pieces_row[6 + BISHOP]
    return popcount(minors) <= 1


@_jit
def is_repetition(
    keys: npt.NDArray[np.uint64], ply: int, game_keys: npt.NDArray[np.uint64], n_game_keys: int
) -> bool:
    """Whether the position at `ply` already occurred on the path or at an earlier root."""
    key = keys[ply]
    for i in range(ply):
        if keys[i] == key:
            return True
    for i in range(n_game_keys):  # noqa: SIM110  (numba wants the plain loop)
        if game_keys[i] == key:
            return True
    return False


@_jit
def score_moves(
    mailbox_row: npt.NDArray[np.int8],
    pieces_row: npt.NDArray[np.uint64],
    stm: int,
    occ: np.uint64,
    moves_row: npt.NDArray[np.int32],
    scores_row: npt.NDArray[np.int64],
    count: int,
    hash_move: int,
    killers_row: npt.NDArray[np.int32],
    history: npt.NDArray[np.int64],
    gain: npt.NDArray[np.int64],
) -> None:
    """Ordering scores: hash move, captures by MVV-LVA (losing ones after the killers),
    killers, then history."""
    for i in range(count):
        move = int(moves_row[i])
        if move == hash_move:
            scores_row[i] = TT_MOVE_BONUS
        elif is_tactical(move):
            victim = victim_type(mailbox_row, move)
            attacker = int(mailbox_row[move & 63] % 6)
            value = 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
            promotion = (move >> 12) & 7
            if promotion:
                value += 10 * PIECE_VALUE[promotion]
            if promotion == 0 and see(pieces_row, mailbox_row, stm, occ, move, gain) < 0:
                scores_row[i] = LOSING_CAPTURE_BONUS + value
            else:
                scores_row[i] = CAPTURE_BONUS + value
        elif move == killers_row[0] or move == killers_row[1]:
            scores_row[i] = KILLER_BONUS
        else:
            scores_row[i] = history[move & 63, (move >> 6) & 63]


@_jit
def pick_next(
    moves_row: npt.NDArray[np.int32], scores_row: npt.NDArray[np.int64], start: int, count: int
) -> int:
    """Swap the best-scored remaining move into `start` and return it."""
    best = start
    for i in range(start + 1, count):
        if scores_row[i] > scores_row[best]:
            best = i
    if best != start:
        moves_row[start], moves_row[best] = moves_row[best], moves_row[start]
        scores_row[start], scores_row[best] = scores_row[best], scores_row[start]
    return int(moves_row[start])


@_jit
def search(
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
    allow_null: bool,
    board: tuple[Any, ...],
    acc: tuple[Any, ...],
    net: tuple[Any, ...],
    tables: tuple[Any, ...],
    ctrl: npt.NDArray[np.int64],
    deadline: float,
) -> int:
    """Negamax with quiescence at depth <= 0. Returns the score for the side to move;
    at the root the best move so far is written to ctrl[CTRL_ROOT_MOVE]."""
    pieces, occupied, mailbox, state, keys = board
    white_acc, black_acc, white_psqt, black_psqt, act, l1c, l1x, l2c, l2x = acc
    ft_w, ft_b, psqt_w, l1_w, l1_b, l2_w, l2_b, out_w, out_b = net
    tt_keys, tt_data, killers, history, moves, scores, game_keys, gain = tables

    ctrl[CTRL_NODES] += 1
    if ctrl[CTRL_NODES] % CLOCK_CHECK_NODES == 0:
        limit = ctrl[CTRL_NODE_LIMIT]
        if (limit > 0 and ctrl[CTRL_NODES] >= limit) or (limit <= 0 and now() > deadline):
            ctrl[CTRL_ABORT] = 1
    if ctrl[CTRL_ABORT]:
        return 0

    stm = int(state[ply, STM])
    occ = occupied[ply, 2]
    key = keys[ply]
    pieces_row = pieces[ply]
    mailbox_row = mailbox[ply]

    if ply > 0 and (
        state[ply, HALFMOVE] >= 100
        or insufficient_material(pieces_row)
        or is_repetition(keys, ply, game_keys, int(ctrl[CTRL_GAME_KEYS]))
    ):
        return 0

    hash_move = -1
    slot = int(key & np.uint64(TT_SIZE - 1))
    if tt_keys[slot] == key:
        data = int(tt_data[slot])
        hash_move = tt_move(data)
        if ply > 0 and tt_depth(data) >= depth:
            stored = from_tt_score(tt_score(data), ply)
            bound = tt_bound(data)
            if bound == EXACT:
                return stored
            if bound == LOWER and stored >= beta:
                return stored
            if bound == UPPER and stored <= alpha:
                return stored
    if ply == 0 and ctrl[CTRL_ROOT_HINT] >= 0:
        hash_move = int(ctrl[CTRL_ROOT_HINT])

    in_check = is_attacked(pieces_row, int(state[ply, WKING + stm]), stm ^ 1, occ)

    # ---- quiescence: captures to a quiet position, so the evaluation is not mid-exchange
    if depth <= 0 or ply >= MAX_PLY_LIMIT:
        if ply >= MAX_PLY_LIMIT:
            return evaluate(
                ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
                act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
            )  # fmt: skip
        static = -2 * MATE
        best = -2 * MATE
        if in_check:
            # No stand-pat while in check: the position may be a forced mate.
            count = generate_moves(pieces, occupied, mailbox, state, ply, moves, False)
        else:
            static = evaluate(
                ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
                act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
            )  # fmt: skip
            if static >= beta:
                return static
            if static > alpha:
                alpha = static
            best = static
            count = generate_moves(pieces, occupied, mailbox, state, ply, moves, True)
        score_moves(
            mailbox_row, pieces_row, stm, occ, moves[ply], scores[ply], count, -1,
            killers[ply], history, gain,
        )  # fmt: skip
        legal = 0
        for i in range(count):
            move = pick_next(moves[ply], scores[ply], i, count)
            if not in_check and (move >> 12) & 7 == 0:
                victim = victim_type(mailbox_row, move)
                if static + VICTIM_VALUE[victim] + DELTA_MARGIN < alpha:
                    continue
                if see(pieces_row, mailbox_row, stm, occ, move, gain) < 0:
                    continue
            if not make_move(pieces, occupied, mailbox, state, keys, ply, move):
                continue
            legal += 1
            update_ply(
                ft_w, ft_b, psqt_w, mailbox, state, ply, move,
                white_acc, black_acc, white_psqt, black_psqt,
            )  # fmt: skip
            score = -search(
                depth - 1, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
            )
            if ctrl[CTRL_ABORT]:
                return 0
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if in_check and legal == 0:
            return -MATE + ply
        return best

    # ---- main search ---------------------------------------------------------------------
    if ply > 0 and depth <= RFP_MAX_DEPTH and not in_check and beta < MATE_THRESHOLD:
        # Reverse futility: a static evaluation comfortably above beta is trusted.
        static = evaluate(
            ply, state, occupied, white_acc, black_acc, white_psqt, black_psqt,
            act, l1c, l1x, l2c, l2x, l1_w, l1_b, l2_w, l2_b, out_w, out_b,
        )  # fmt: skip
        if static - RFP_MARGIN * depth >= beta:
            return static

    if (
        ply > 0
        and allow_null
        and depth >= NULL_MIN_DEPTH
        and not in_check
        and beta < MATE_THRESHOLD
        and has_major_material(pieces_row, stm)
    ):
        make_null(pieces, occupied, mailbox, state, keys, ply)
        copy_ply(ply, white_acc, black_acc, white_psqt, black_psqt)
        passed = -search(
            depth - 1 - NULL_REDUCTION, ply + 1, -beta, -beta + 1, False,
            board, acc, net, tables, ctrl, deadline,
        )  # fmt: skip
        if ctrl[CTRL_ABORT]:
            return 0
        if passed >= beta:
            # Deliberately beta, not `passed`: a score borrowed from a position that
            # skipped a turn is not trustworthy enough to store as the real value.
            return beta

    count = generate_moves(pieces, occupied, mailbox, state, ply, moves, False)
    score_moves(
        mailbox_row, pieces_row, stm, occ, moves[ply], scores[ply], count, hash_move,
        killers[ply], history, gain,
    )  # fmt: skip

    original_alpha = alpha
    best = -2 * MATE
    best_move = -1
    legal = 0
    for i in range(count):
        move = pick_next(moves[ply], scores[ply], i, count)
        if not make_move(pieces, occupied, mailbox, state, keys, ply, move):
            continue
        update_ply(
            ft_w, ft_b, psqt_w, mailbox, state, ply, move,
            white_acc, black_acc, white_psqt, black_psqt,
        )  # fmt: skip
        index = legal
        legal += 1
        tactical = is_tactical(move)
        gives_check = is_attacked(
            pieces[ply + 1], int(state[ply + 1, WKING + (stm ^ 1)]), stm, occupied[ply + 1, 2]
        )
        # Checks are forcing: a line of them is cheap to follow and expensive to cut
        # short, so a checking move is searched one ply deeper.
        child = depth - 1 + (1 if gives_check else 0)
        reduction = 0
        if (
            depth >= LMR_MIN_DEPTH
            and index >= LMR_MIN_MOVES
            and not tactical
            and not in_check
            and not gives_check
        ):
            reduction = 2 if (index >= LMR_LATE_MOVES and depth >= LMR_DEEP) else 1
        if index == 0:
            score = -search(
                child, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
            )
        else:
            # Principal variation search: later moves first answer "better than alpha?"
            # with a null window; only a yes earns a full-window search.
            score = -search(
                child - reduction, ply + 1, -alpha - 1, -alpha, True,
                board, acc, net, tables, ctrl, deadline,
            )  # fmt: skip
            if reduction and score > alpha and not ctrl[CTRL_ABORT]:
                score = -search(
                    child, ply + 1, -alpha - 1, -alpha, True,
                    board, acc, net, tables, ctrl, deadline,
                )  # fmt: skip
            if alpha < score < beta and not ctrl[CTRL_ABORT]:
                score = -search(
                    child, ply + 1, -beta, -alpha, True, board, acc, net, tables, ctrl, deadline
                )
        if ctrl[CTRL_ABORT]:
            return 0
        if score > best:
            best = score
            best_move = move
            if ply == 0:
                ctrl[CTRL_ROOT_MOVE] = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if not tactical:
                if killers[ply, 0] != move:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                history[move & 63, (move >> 6) & 63] += depth * depth
            break

    if legal == 0:
        return -MATE + ply if in_check else 0

    if best <= original_alpha:
        bound = UPPER
    elif best >= beta:
        bound = LOWER
    else:
        bound = EXACT
    existing = int(tt_data[slot])
    keep = (
        tt_keys[slot] == key
        and tt_generation(existing) == (int(ctrl[CTRL_GENERATION]) & 15)
        and tt_depth(existing) > depth
    )
    if not keep:
        tt_keys[slot] = key
        tt_data[slot] = pack_tt(
            depth, bound, int(ctrl[CTRL_GENERATION]), max(best_move, 0), to_tt_score(best, ply)
        )
    return best
