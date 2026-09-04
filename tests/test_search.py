"""Search behaviour: time budget, draw detection, move ordering, and move choice.

These tests pin the properties that lose games when they break — the clock, legality,
and the draw rules — rather than exact scores, which depend on the trained net.
"""

import chess

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
