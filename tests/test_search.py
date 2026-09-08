"""Behavioural tests for the bitboard search.

Strength is judged by SPRT, not here; these pin the rules: legality, mates, draws, the
clock, and the table's bookkeeping.
"""

import random
import time

import chess
import numpy as np
import pytest

import bitboard as bb
import search
import search_kernel as sk
from search import Searcher
from tools.gen_random_net import random_network

_NET = random_network(2026)


@pytest.fixture(scope="module")
def searcher() -> Searcher:
    s = Searcher(_NET)
    s.warm_up()
    return s


def test_budget_scales_with_clock() -> None:
    assert search.budget_ms(120_000) > search.budget_ms(12_000)


def test_budget_never_exceeds_fraction_of_clock() -> None:
    assert search.budget_ms(100) <= search.MAX_FRACTION * 100


def test_budget_stays_positive_on_a_nearly_dead_clock() -> None:
    assert search.budget_ms(1) >= search.MIN_BUDGET_MS


def test_hard_budget_extends_the_soft_one_within_the_clock_share() -> None:
    assert search.hard_budget_ms(120_000) > search.budget_ms(120_000)
    assert search.hard_budget_ms(120_000) <= search.HARD_FACTOR * search.budget_ms(120_000)
    assert search.hard_budget_ms(1_000) <= search.MAX_FRACTION * 1_000
    assert search.hard_budget_ms(1) >= search.budget_ms(1)


def test_instability_means_a_changed_move_or_a_falling_score() -> None:
    assert not search.unstable(-1, 5, None, 100)
    assert not search.unstable(5, 5, 100, 100)
    assert search.unstable(5, 6, 100, 100)
    assert search.unstable(5, 5, 100, 100 - search.SCORE_DROP - 1)
    assert not search.unstable(5, 5, 100, 100 - search.SCORE_DROP + 1)


def test_shallow_iterations_never_count_as_unstable() -> None:
    assert not search.unstable(5, 6, 100, 100, depth=search.UNSTABLE_MIN_DEPTH - 1)
    assert search.unstable(5, 6, 100, 100, depth=search.UNSTABLE_MIN_DEPTH)


def test_horizon_shrinks_with_the_move_number_down_to_a_floor() -> None:
    assert search.horizon(1) > search.horizon(20) > search.horizon(30)
    assert search.horizon(60) == search.HORIZON_MIN
    assert search.horizon(200) == search.HORIZON_MIN


def test_budget_is_flat_through_the_middlegame() -> None:
    # The clocks a rated game actually had at moves 5, 20 and 35: the old fixed horizon
    # spent 5.2 s, 2.6 s and 1.6 s on them; the sharp moves at 35 need more than that.
    early = search.budget_ms(115_000, 5)
    middle = search.budget_ms(58_000, 20)
    late = search.budget_ms(31_000, 35)
    assert 3_000 <= early <= 4_500
    assert middle >= 0.8 * early
    assert late >= 0.5 * early


def test_budget_never_flags_over_a_long_game() -> None:
    clock = 120_000.0
    for fullmove in range(1, 120):
        spend = search.hard_budget_ms(int(clock), fullmove) + search.SAFETY_MS
        assert spend < clock
        clock = clock - spend + search.INCREMENT_MS
    assert clock > 0


def test_mate_scores_are_stored_relative_to_the_node() -> None:
    assert search.to_tt_score(sk.MATE - 5, 3) == sk.MATE - 2
    assert search.from_tt_score(sk.MATE - 2, 3) == sk.MATE - 5
    assert search.to_tt_score(-sk.MATE + 5, 3) == -sk.MATE + 2
    assert search.to_tt_score(1234, 3) == 1234


def test_tt_packing_round_trips() -> None:
    move = bb.encode_move(12, 28, 0, bb.FLAG_DOUBLE)
    data = sk.pack_tt(12, sk.LOWER, 9, move, -123456)
    assert sk.tt_depth(data) == 12
    assert sk.tt_bound(data) == sk.LOWER
    assert sk.tt_generation(data) == 9
    assert sk.tt_move(data) == move
    assert sk.tt_score(data) == -123456


def test_lmr_table_grows_with_depth_and_move_number() -> None:
    assert sk.LMR_TABLE[3, 4] == 1
    assert sk.LMR_TABLE[9, 10] == 2
    assert sk.LMR_TABLE[12, 20] == 4
    for depth in range(3, 20):
        for index in range(4, 60):
            assert sk.LMR_TABLE[depth, index] <= sk.LMR_TABLE[depth + 1, index]
            assert sk.LMR_TABLE[depth, index] <= sk.LMR_TABLE[depth, index + 1]


def test_pruning_limits_are_ordered() -> None:
    for depth in range(1, sk.LMP_MAX_DEPTH):
        assert sk.lmp_limit(depth, True) <= sk.lmp_limit(depth + 1, True)
        assert sk.lmp_limit(depth, False) <= sk.lmp_limit(depth, True)
    assert sk.lmp_limit(1, False) >= 2  # never prune the first reply
    assert sk.lmp_limit(sk.LMP_MAX_DEPTH, True) > sk.LMR_MIN_MOVES
    assert sk.FUTILITY_MAX_DEPTH < sk.RFP_MAX_DEPTH + 1


def test_aspiration_window_brackets_the_previous_score() -> None:
    alpha, beta, window = search.aspiration_window(1000, search.ASPIRATION_MIN_DEPTH)
    assert alpha == 1000 - window
    assert beta == 1000 + window
    assert search.aspiration_window(1000, 1)[0] == -2 * sk.MATE


def test_widen_opens_only_the_failed_side() -> None:
    alpha, beta, window = search.widen(900, 1100, 850, 100)
    assert (alpha, beta, window) == (850 - 200, 1100, 200)


def _see_of(searcher: Searcher, fen: str, uci: str) -> int:
    board = chess.Board(fen)
    bb.set_from_board(board, searcher.board, 0)
    move = chess.Move.from_uci(uci)
    flags = bb.FLAG_CAPTURE if board.is_capture(move) else 0
    if board.is_en_passant(move):
        flags |= bb.FLAG_EP
    encoded = bb.encode_move(move.from_square, move.to_square, 0, flags)
    return int(
        sk.see(
            searcher.board.pieces[0],
            searcher.board.mailbox[0],
            int(searcher.board.state[0, bb.STM]),
            searcher.board.occupied[0, 2],
            encoded,
            searcher.see_gain,
        )
    )


def test_see_scores_a_free_pawn_and_a_defended_one(searcher: Searcher) -> None:
    assert _see_of(searcher, "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 1
    assert _see_of(searcher, "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == 0


def test_see_lets_the_defender_decline_a_losing_recapture(searcher: Searcher) -> None:
    # NxP defended by a pawn: white loses the knight for a pawn.
    assert _see_of(searcher, "4k3/8/2p5/3p4/4N3/8/8/4K3 w - - 0 1", "e4d5") == -2


def test_see_sees_the_rook_behind_the_rook(searcher: Searcher) -> None:
    # RxP on d5 defended by a rook; a second white rook behind the first recaptures.
    assert _see_of(searcher, "3rk3/8/8/3p4/8/8/3R4/3R2K1 w - - 0 1", "d2d5") == 1


def test_see_handles_en_passant(searcher: Searcher) -> None:
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"
    assert _see_of(searcher, fen, "e5d6") == 1


def test_finds_mate_in_one(searcher: Searcher) -> None:
    assert searcher.pick("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 2000).uci() == "a1a8"


def test_finds_mate_in_two_through_a_quiet_first_move(searcher: Searcher) -> None:
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    assert searcher.pick(fen, 3000).uci() == "h5f7"


def test_takes_the_free_queen(searcher: Searcher) -> None:
    fen = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    assert searcher.pick(fen, 1500).uci() == "f3h4"


def test_returns_a_legal_move_when_in_check(searcher: Searcher) -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/7P/5P2/PPPPP1Pq/RNBQKBNR w KQkq - 1 3")
    assert searcher.pick(board.fen(), 1000) in board.legal_moves


def test_never_returns_an_illegal_move(searcher: Searcher) -> None:
    rng = random.Random(8)
    for _game in range(40):
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over():
                break
            if rng.random() < 0.3:
                move = searcher.pick(board.fen(), 300)
                assert move in board.legal_moves, board.fen()
            else:
                move = rng.choice(list(board.legal_moves))
            board.push(move)


def test_respects_a_tight_time_budget(searcher: Searcher) -> None:
    start = time.monotonic()
    searcher.pick("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 1000)
    assert time.monotonic() - start < 0.5


def test_node_limit_makes_the_search_deterministic(searcher: Searcher) -> None:
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    searcher.warm_up()
    first = searcher.pick(fen, 60_000, node_limit=20_000)
    first_nodes = searcher.nodes
    searcher.warm_up()
    second = searcher.pick(fen, 60_000, node_limit=20_000)
    assert first == second
    assert first_nodes == searcher.nodes


def test_no_legal_moves_raises(searcher: Searcher) -> None:
    with pytest.raises(ValueError):
        searcher.pick("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1000)


def test_root_position_is_remembered_for_repetition(searcher: Searcher) -> None:
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    searcher.note_root_position(board)
    stacks = bb.new_stacks()
    bb.set_from_board(board, stacks, 0)
    assert searcher.is_draw_key(stacks.keys[0])


def test_history_resets_when_the_fullmove_number_goes_backwards(searcher: Searcher) -> None:
    searcher.warm_up()
    searcher.note_root_position(chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 40"))
    assert searcher.ctrl[sk.CTRL_GAME_KEYS] == 1
    searcher.note_root_position(chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 3"))
    assert searcher.ctrl[sk.CTRL_GAME_KEYS] == 1


def test_position_after_our_move_is_remembered(searcher: Searcher) -> None:
    searcher.warm_up()
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    move = searcher.pick(board.fen(), 500)
    board.push(move)
    stacks = bb.new_stacks()
    bb.set_from_board(board, stacks, 0)
    assert searcher.is_draw_key(stacks.keys[0])


def test_never_lets_a_repetition_claim_open_while_winning(searcher: Searcher) -> None:
    # K+Q vs K against a shuffling king. The platform ends a game the moment the side
    # to move could claim a threefold, so no position may ever become claimable after
    # one of our moves, and the game must still be converted.
    searcher.warm_up()
    board = chess.Board("7k/8/8/8/8/8/8/K6Q w - - 0 1")
    for _ in range(60):
        if board.is_game_over():
            break
        board.push(searcher.pick(board.fen(), 1500))
        assert not board.can_claim_threefold_repetition(), board.fen()
        if board.is_game_over():
            break
        board.push(min(board.legal_moves, key=lambda m: m.uci()))
        assert not board.can_claim_threefold_repetition(), board.fen()
    assert board.is_checkmate(), board.fen()


def test_agent_returns_legal_uci() -> None:
    import agent

    board = chess.Board()
    assert chess.Move.from_uci(agent.get_move(board.fen(), 5000)) in board.legal_moves


def test_agent_falls_back_rather_than_raising_on_a_dead_clock() -> None:
    import agent

    move = chess.Move.from_uci(agent.get_move(chess.STARTING_FEN, 0))
    assert move in chess.Board().legal_moves


def _adjusted(searcher: Searcher, fen: str, halfmove: int | None = None) -> int:
    board = chess.Board(fen)
    if halfmove is not None:
        board.halfmove_clock = halfmove
    bb.set_from_board(board, searcher.board, 0)
    stm = int(searcher.board.state[0, bb.STM])
    return int(sk.adjust_eval(1000, searcher.board.pieces[0], searcher.board.state[0], stm))


def test_score_fades_with_the_fifty_move_counter(searcher: Searcher) -> None:
    fen = "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    fresh = _adjusted(searcher, fen, 0)
    stale = _adjusted(searcher, fen, 90)
    assert fresh == 1000
    assert 0 < stale < 0.4 * fresh


def test_mop_up_pays_for_cornering_the_bare_king(searcher: Searcher) -> None:
    corner = _adjusted(searcher, "7k/8/6K1/8/8/8/8/R7 w - - 0 1")
    centre = _adjusted(searcher, "8/8/3k4/8/8/8/8/R3K3 w - - 0 1")
    assert corner > centre > 1000


def test_mop_up_only_applies_when_the_loser_is_bare(searcher: Searcher) -> None:
    assert _adjusted(searcher, "7k/7p/6K1/8/8/8/8/R7 w - - 0 1") == 1000
    assert _adjusted(searcher, "7k/8/6K1/8/8/8/8/R6r w - - 0 1") == 1000


def test_mop_up_is_symmetric_for_the_losing_side_to_move(searcher: Searcher) -> None:
    board = chess.Board("7k/8/6K1/8/8/8/8/R7 b - - 0 1")
    bb.set_from_board(board, searcher.board, 0)
    stm = int(searcher.board.state[0, bb.STM])
    score = int(sk.adjust_eval(-1000, searcher.board.pieces[0], searcher.board.state[0], stm))
    assert score < -1000


def test_history_stays_bounded_and_decays() -> None:
    history = np.zeros((64, 64), dtype=np.int64)
    move = 12 | (28 << 6)  # e2e4 encoded as from | to << 6
    for _ in range(200):
        sk.history_update(history, move, sk.history_bonus(9))
    assert 0 < history[12, 28] <= sk.HISTORY_MAX
    peak = int(history[12, 28])
    for _ in range(20):
        sk.history_update(history, move, -sk.history_bonus(9))
    assert history[12, 28] < peak
    assert sk.history_bonus(1) < sk.history_bonus(4) <= sk.HISTORY_BONUS_CAP
    assert sk.HISTORY_MAX // sk.HISTORY_LMR_DIVISOR <= 2
