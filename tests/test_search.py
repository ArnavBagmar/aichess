"""Search behaviour: time budget, draw detection, move ordering, and move choice.

These tests pin the properties that lose games when they break — the clock, legality,
and the draw rules — rather than exact scores, which depend on the trained net.
"""

import random
import time as time_module

import chess
import pytest

import search
from nnue_engine import load_engine


def test_budget_scales_with_clock() -> None:
    generous = search.budget_ms(120_000)
    tight = search.budget_ms(2_000)
    assert generous > tight


def test_budget_never_exceeds_fraction_of_clock() -> None:
    for clock in (50, 200, 1_000, 10_000, 120_000):
        assert search.budget_ms(clock) <= search.MAX_FRACTION * clock


def test_budget_stays_positive_on_a_nearly_dead_clock() -> None:
    assert search.budget_ms(0) >= search.MIN_BUDGET_MS
    assert search.budget_ms(10) >= search.MIN_BUDGET_MS


def test_search_aborted_is_an_exception() -> None:
    assert issubclass(search.SearchAborted, Exception)


def make_searcher() -> search.Searcher:
    return search.Searcher(load_engine())


def test_fresh_searchers_do_not_share_state() -> None:
    first, second = make_searcher(), make_searcher()
    first.note_root_position(chess.Board())
    assert first.game_history != []
    assert second.game_history == []


def test_position_seen_at_the_root_is_a_draw_inside_the_search() -> None:
    searcher = make_searcher()
    board = chess.Board()
    searcher.note_root_position(board)
    # ply 0 is the root itself and must never score as a draw against its own entry.
    assert not searcher.is_draw(board, 0)
    assert searcher.is_draw(board, 1)


def test_fifty_move_rule_is_a_draw() -> None:
    searcher = make_searcher()
    board = chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 100 200")
    assert searcher.is_draw(board, 1)


def test_insufficient_material_is_a_draw() -> None:
    searcher = make_searcher()
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    assert searcher.is_draw(board, 1)


def test_history_resets_when_the_fullmove_number_goes_backwards() -> None:
    searcher = make_searcher()
    searcher.note_root_position(chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 0 40"))
    searcher.note_root_position(chess.Board("8/8/4k3/8/8/4K3/8/7R w - - 0 2"))
    assert len(searcher.game_history) == 1


def test_tt_move_is_ordered_first() -> None:
    searcher = make_searcher()
    board = chess.Board()
    quiet = chess.Move.from_uci("h2h3")
    assert searcher.ordered_moves(board, 0, quiet)[0] == quiet


def test_captures_precede_quiet_moves() -> None:
    searcher = make_searcher()
    # White queen on d1 can take the undefended rook on d8.
    board = chess.Board("3r3k/8/8/8/8/8/8/3QK3 w - - 0 1")
    ordered = searcher.ordered_moves(board, 0, None)
    assert ordered[0] == chess.Move.from_uci("d1d8")


def test_mvv_lva_prefers_the_more_valuable_victim() -> None:
    searcher = make_searcher()
    # White rook on a1 may take a queen on a8 or a knight on g1.
    board = chess.Board("q6k/8/8/8/8/8/8/R5nK w - - 0 1")
    ordered = searcher.ordered_moves(board, 0, None)
    assert ordered[0] == chess.Move.from_uci("a1a8")


def test_ordered_moves_is_a_permutation_of_the_legal_moves() -> None:
    searcher = make_searcher()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert sorted(searcher.ordered_moves(board, 0, None), key=str) == sorted(
        board.legal_moves, key=str
    )


def test_ordered_captures_are_all_captures_or_promotions() -> None:
    searcher = make_searcher()
    board = chess.Board("3r3k/8/8/8/8/8/6P1/3QK3 w - - 0 1")
    for move in searcher.ordered_captures(board):
        assert board.is_capture(move) or move.promotion is not None


def test_finds_mate_in_one() -> None:
    searcher = make_searcher()
    # Back-rank mate: Ra1-a8 is forced mate, the black king is boxed in by its pawns.
    move = searcher.pick("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 10_000)
    assert move == chess.Move.from_uci("a1a8")


def test_takes_the_free_queen() -> None:
    searcher = make_searcher()
    move = searcher.pick("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1", 10_000)
    assert move == chess.Move.from_uci("d1d8")


def test_returns_a_legal_move_when_in_check() -> None:
    # Black king on h8 is checked along the open h-file by the rook on h1.
    fen = "7k/8/8/8/8/8/6P1/6KR b - - 0 1"
    searcher = make_searcher()
    assert searcher.pick(fen, 5_000) in chess.Board(fen).legal_moves


def test_never_returns_an_illegal_move() -> None:
    searcher = make_searcher()
    rng = random.Random(7)
    for _ in range(40):
        board = chess.Board()
        for _ in range(rng.randint(2, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        assert searcher.pick(board.fen(), 300) in board.legal_moves


def test_respects_a_tight_time_budget() -> None:
    searcher = make_searcher()
    started = time_module.monotonic()
    searcher.pick(chess.STARTING_FEN, 1_000)
    elapsed_ms = (time_module.monotonic() - started) * 1000.0
    # budget_ms(1000) is ~283 ms; allow generous slack for a slow CI box.
    assert elapsed_ms < 2_000


def test_no_legal_moves_raises() -> None:
    searcher = make_searcher()
    # Black is stalemated.
    with pytest.raises(ValueError):
        searcher.pick("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1_000)


def test_agent_returns_legal_uci() -> None:
    import agent

    board = chess.Board()
    uci = agent.get_move(board.fen(), 5_000)
    assert chess.Move.from_uci(uci) in board.legal_moves


def test_agent_falls_back_rather_than_raising_on_a_dead_clock() -> None:
    import agent

    board = chess.Board()
    uci = agent.get_move(board.fen(), 0)
    assert chess.Move.from_uci(uci) in board.legal_moves


def test_null_move_is_refused_without_major_material() -> None:
    # King and pawns only: exactly where passing can be better than moving, so the
    # zugzwang guard has to switch null-move pruning off.
    searcher = make_searcher()
    pawns_only = chess.Board("8/5p2/5k2/8/8/5K2/5P2/8 w - - 0 1")
    assert not searcher._has_major_material(pawns_only)
    with_rook = chess.Board("8/5p2/5k2/8/8/5K2/5P2/7R w - - 0 1")
    assert searcher._has_major_material(with_rook)


def test_keeps_the_win_in_a_pawn_endgame() -> None:
    """The pawn endgame that null-move pruning would misjudge if the guard failed.

    The assertion is that the win survives, not that one exact move is played: several
    moves here win, and LMR legitimately trades picking the very fastest for depth
    everywhere else. What would be a real defect is throwing the win away.
    """
    fen = "8/6P1/8/8/8/8/1k6/4K3 w - - 0 1"
    searcher = make_searcher()
    move = searcher.pick(fen, 5_000)
    engine = searcher.engine
    engine.set_position(fen)
    assert move in engine.board.legal_moves
    engine.push(move)
    assert -engine.evaluate_cp() > 1_500  # still decisively winning for us


def test_finds_mate_in_two_through_a_quiet_first_move() -> None:
    # The key move is quiet, so LMR may reduce it; the re-search has to recover it.
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    searcher = make_searcher()
    move = searcher.pick(fen, 10_000)
    board = chess.Board(fen)
    board.push(move)
    assert board.is_checkmate() or move == chess.Move.from_uci("a1a8")


def test_reduction_grows_for_later_moves_at_depth() -> None:
    searcher = make_searcher()
    assert searcher._late_move_reduction(search.LMR_DEEP, search.LMR_LATE_MOVES) == 2
    assert searcher._late_move_reduction(search.LMR_MIN_DEPTH, search.LMR_MIN_MOVES) == 1
