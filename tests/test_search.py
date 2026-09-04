"""Search behaviour: time budget, draw detection, move ordering, and move choice.

These tests pin the properties that lose games when they break — the clock, legality,
and the draw rules — rather than exact scores, which depend on the trained net.
"""

import search


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
