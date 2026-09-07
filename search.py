"""Root of the search: time budget, iterative deepening, aspiration windows, and the
python-chess safety check around the numba kernel in search_kernel.py.

The Searcher owns every array the kernel uses and lives for one game, so the table,
killers, history and the record of root positions accumulate across our moves and can
never leak into another game. Scores stay in the net's 1/32-centipawn integer units.
"""

import time
from pathlib import Path
from typing import Final

import chess
import numpy as np

import bitboard as bb
import nnue_bitboard as nb
from nnue_arch import SCORE_PER_CP, WEIGHTS_FILE
from nnue_net import NetworkWeights, load_network
from search_kernel import (
    CTRL_ABORT,
    CTRL_GAME_KEYS,
    CTRL_GENERATION,
    CTRL_NODE_LIMIT,
    CTRL_NODES,
    CTRL_ROOT_DEPTH,
    CTRL_ROOT_HINT,
    CTRL_ROOT_MOVE,
    CTRL_SIZE,
    MATE,
    MATE_THRESHOLD,
    MAX_DEPTH,
    MAX_PLY_LIMIT,
    TT_SIZE,
    search,
)
from search_kernel import from_tt_score as _from_tt_score
from search_kernel import to_tt_score as _to_tt_score

# Time control. Every term but INCREMENT_MS comes from the clock we were handed.
# A fixed horizon of N moves spends time_left/N, which decays with the clock: rated games
# showed 5-11 s on the first moves and 1.2-1.6 s on the sharp middlegame moves that lost
# them, with 20-30 s still unused at the end. The horizon now follows the move number
# (the opening has many moves ahead, the middlegame fewer) with a floor that lets the
# increment carry long endgames, which flattens the spend to ~3.7 s through move 30.
HORIZON_BASE: Final = 40  # moves remaining assumed at move 0
HORIZON_MIN: Final = 16  # never plan for fewer moves than this
HARD_FACTOR: Final = 2.0  # an unstable search may spend this many budgets
UNSTABLE_MIN_DEPTH: Final = 6  # shallow iterations flip constantly; ignore them
SCORE_DROP: Final = 40 * SCORE_PER_CP  # a fall this large marks the search unstable
INCREMENT_MS: Final = 500  # published time control: 120 s + 0.5 s per move
SAFETY_MS: Final = 50  # margin; the referee measures wall time and does not forgive
MAX_FRACTION: Final = 0.4  # never spend more than this much of what is left
MIN_BUDGET_MS: Final = 10  # always attempt something

# Aspiration windows. From ASPIRATION_MIN_DEPTH on, the root searches a narrow window
# around the previous iteration's score; a fail outside it widens that side and retries.
ASPIRATION_MIN_DEPTH: Final = 5
ASPIRATION_WINDOW: Final = 250 * SCORE_PER_CP  # about 1.3 pawns on the net's scale

MAX_GAME_KEYS: Final = 1024  # root positions remembered for repetition detection


def horizon(fullmove: int) -> int:
    """Moves we plan to spread the remaining clock over, given the move number."""
    return max(HORIZON_MIN, HORIZON_BASE - fullmove)


def budget_ms(time_left_ms: int, fullmove: int = 1) -> float:
    """Milliseconds to spend on this move, derived from the clock we were handed.

    Only a fraction of INCREMENT_MS is claimed: it is the one number taken from the
    published rules rather than the input, so if the platform ever changed the
    increment the result is a conservative budget rather than a flag.
    """
    budget = time_left_ms / horizon(fullmove) + 0.6 * INCREMENT_MS
    budget = min(budget, MAX_FRACTION * time_left_ms)
    return max(budget - SAFETY_MS, MIN_BUDGET_MS)


def hard_budget_ms(time_left_ms: int, fullmove: int = 1) -> float:
    """The most an unstable search may spend: several budgets, never a big clock share."""
    soft = budget_ms(time_left_ms, fullmove)
    return max(min(HARD_FACTOR * soft, MAX_FRACTION * time_left_ms - SAFETY_MS), soft)


def unstable(
    previous_move: int, move: int, previous_score: int | None, score: int, depth: int = MAX_DEPTH
) -> bool:
    """Whether the last iteration changed its mind or saw the position get worse.

    Iterations shallower than UNSTABLE_MIN_DEPTH do not count: they change their mind
    on nearly every move, and extending on them is what burned 5-11 s per opening move.
    """
    if depth < UNSTABLE_MIN_DEPTH:
        return False
    if previous_move >= 0 and move >= 0 and move != previous_move:
        return True
    return previous_score is not None and score < previous_score - SCORE_DROP


def to_tt_score(score: int, ply: int) -> int:
    """Root-relative mate score to node-relative, for storage."""
    return int(_to_tt_score(score, ply))


def from_tt_score(score: int, ply: int) -> int:
    """Inverse of to_tt_score, for a probe at `ply`."""
    return int(_from_tt_score(score, ply))


def aspiration_window(score: int, depth: int) -> tuple[int, int, int]:
    """(alpha, beta, window) to open iteration `depth` with, given the last score."""
    if depth < ASPIRATION_MIN_DEPTH:
        return -2 * MATE, 2 * MATE, 2 * MATE
    return score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW, ASPIRATION_WINDOW


def widen(alpha: int, beta: int, score: int, window: int) -> tuple[int, int, int]:
    """Open the side the search failed on, doubling the window, up to the full range."""
    window *= 2
    if score <= alpha:
        alpha = max(score - window, -2 * MATE)
    else:
        beta = min(score + window, 2 * MATE)
    return alpha, beta, window


class Searcher:
    """Alpha-beta search state for one game."""

    def __init__(self, net: NetworkWeights) -> None:
        self.net = net
        self.board = bb.new_stacks()
        self._after = bb.new_stacks()  # scratch for the position our move creates
        self.acc = nb.new_acc_stacks()
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_data = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.full((MAX_PLY_LIMIT + 2, 2), -1, dtype=np.int32)
        self.history = np.zeros((64, 64), dtype=np.int64)
        self.moves = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int32)
        self.scores = np.zeros((bb.MAX_PLY, bb.MAX_MOVES), dtype=np.int64)
        self.game_keys = np.zeros(MAX_GAME_KEYS, dtype=np.uint64)
        self.see_gain = np.zeros(32, dtype=np.int64)
        self.null_flags = np.zeros(bb.MAX_PLY, dtype=np.int8)  # 1 where a null move was made
        self.ctrl = np.zeros(CTRL_SIZE, dtype=np.int64)
        self.nodes = 0
        self.score = 0
        self._last_fullmove = 0
        self._net_tuple = (
            net.ft_w,
            net.ft_b,
            net.psqt_w,
            net.l1_w,
            net.l1_b,
            net.l2_w,
            net.l2_b,
            net.out_w,
            net.out_b,
        )
        self._tables = (
            self.tt_keys,
            self.tt_data,
            self.killers,
            self.history,
            self.moves,
            self.scores,
            self.game_keys,
            self.see_gain,
            self.null_flags,
        )

    def note_root_position(self, board: chess.Board) -> None:
        """Record a position we were asked about, for repetition detection.

        We only ever observe positions where it is our turn, every other ply, which is
        enough: a position repeating at our turn is a genuine repetition.
        """
        if board.fullmove_number < self._last_fullmove:
            # Cannot follow the previous root. Never expected within one game; this
            # exists so a surprise cannot become a bogus draw claim.
            self.ctrl[CTRL_GAME_KEYS] = 0
        self._last_fullmove = board.fullmove_number
        bb.set_from_board(board, self.board, 0)
        self._remember(self.board.keys[0])

    def note_move_played(self, board: chess.Board, move: chess.Move) -> None:
        """Record the position our own move creates.

        The platform ends a game as soon as the side to move could claim a repetition,
        so a position we have created twice is a draw waiting to happen even if we
        never repeat it a third time. Six rated draws in won positions were exactly
        that: the engine shuffled through positions after its own moves, which were
        never in its history, until a claim opened up on its turn.
        """
        after = board.copy(stack=False)
        after.push(move)
        bb.set_from_board(after, self._after, 0)
        self._remember(self._after.keys[0])

    def _remember(self, key: np.uint64) -> None:
        n = int(self.ctrl[CTRL_GAME_KEYS])
        if n < MAX_GAME_KEYS:
            self.game_keys[n] = key
            self.ctrl[CTRL_GAME_KEYS] = n + 1

    def is_draw_key(self, key: np.uint64) -> bool:
        """Whether a position key was seen at an earlier root of this game."""
        n = int(self.ctrl[CTRL_GAME_KEYS])
        return bool(np.any(self.game_keys[:n] == key))

    def _set_root(self, board: chess.Board) -> None:
        bb.set_from_board(board, self.board, 0)
        nb.refresh_ply(
            self.net.ft_w,
            self.net.ft_b,
            self.net.psqt_w,
            self.board.mailbox,
            self.board.state,
            0,
            self.acc.white_acc,
            self.acc.black_acc,
            self.acc.white_psqt,
            self.acc.black_psqt,
        )

    def _search(
        self, depth: int, alpha: int, beta: int, hint: int, deadline: float
    ) -> tuple[int, int]:
        self.ctrl[CTRL_ROOT_MOVE] = -1
        self.ctrl[CTRL_ROOT_HINT] = hint
        self.ctrl[CTRL_ROOT_DEPTH] = depth
        score = search(
            depth,
            0,
            alpha,
            beta,
            tuple(self.board),
            tuple(self.acc),
            self._net_tuple,
            self._tables,
            self.ctrl,
            deadline,
        )
        return int(score), int(self.ctrl[CTRL_ROOT_MOVE])

    def pick(self, fen: str, time_left_ms: int, node_limit: int = 0) -> chess.Move:
        """Best move for `fen` within the budget implied by `time_left_ms`.

        A positive `node_limit` caps the search by nodes instead of the clock, which
        makes a search reproducible for tests and benchmarks.
        """
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(f"no legal moves in {fen!r}")
        self.note_root_position(board)
        self._set_root(board)
        started = time.monotonic()
        soft = started + budget_ms(time_left_ms, board.fullmove_number) / 1000.0
        hard = started + hard_budget_ms(time_left_ms, board.fullmove_number) / 1000.0
        self.ctrl[CTRL_NODES] = 0
        self.ctrl[CTRL_ABORT] = 0
        self.ctrl[CTRL_NODE_LIMIT] = node_limit
        self.ctrl[CTRL_GENERATION] += 1

        best = -1
        score = 0
        previous_move = -1
        previous_score: int | None = None
        extend = False
        for depth in range(1, MAX_DEPTH):
            deadline = hard if extend else soft
            if depth > 1 and time.monotonic() >= deadline:
                break
            alpha, beta, window = aspiration_window(score, depth)
            move = -1
            while True:
                value, move = self._search(depth, alpha, beta, best, deadline)
                if self.ctrl[CTRL_ABORT]:
                    break  # discard this depth entirely; it has a biased best move
                if alpha < value < beta or (alpha == -2 * MATE and beta == 2 * MATE):
                    score = value
                    break
                if value >= beta and move >= 0:
                    # A fail-high names a move that beat the window: the best lead we
                    # have if the clock cuts the re-search short. A fail-low names
                    # nothing: null-window scores are not comparable.
                    best = move
                alpha, beta, window = widen(alpha, beta, value, window)
            if self.ctrl[CTRL_ABORT]:
                break
            if move >= 0:
                best = move
            if score >= MATE_THRESHOLD:
                break  # a forced mate for us is as good as it gets
            # A forced mate against us is not final: deeper iterations find the longest
            # defence, and the opponent has to see the whole line to cash it.
            extend = unstable(previous_move, move, previous_score, score, depth)
            previous_move, previous_score = move, score
        self.nodes = int(self.ctrl[CTRL_NODES])
        self.score = score  # last completed iteration, for the side to move, 1/32 cp

        chosen = legal[0]
        if best >= 0:
            proposed = bb.move_to_chess(best)
            if proposed in legal:
                chosen = proposed
            else:
                print(
                    f"kernel proposed illegal {proposed.uci()} in {fen!r}; playing the first legal"
                )
        self.note_move_played(board, chosen)
        return chosen

    def warm_up(self) -> None:
        """Compile every kernel inside the import budget, then leave no state behind."""
        self.pick(chess.STARTING_FEN, 200)
        self.pick("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 200)
        self.tt_keys[:] = 0
        self.tt_data[:] = 0
        self.killers[:] = -1
        self.history[:] = 0
        self.ctrl[:] = 0
        self._last_fullmove = 0


def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / WEIGHTS_FILE


def load_searcher(path: Path | None = None) -> Searcher:
    return Searcher(load_network(path if path is not None else default_weights_path()))
