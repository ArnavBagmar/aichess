"""The bench's arithmetic and scheduling, tested without launching a single game."""

import pytest

from tools.elo_bench import (
    Sprt,
    Tally,
    describe_stockfish,
    expected_score,
    log_likelihood_ratio,
    stockfish_limit,
    verdict,
)


def test_sprt_bounds_are_symmetric_at_equal_error_rates() -> None:
    sprt = Sprt()
    assert sprt.lower == pytest.approx(-2.944, abs=0.001)
    assert sprt.upper == pytest.approx(2.944, abs=0.001)


def test_expected_score_follows_the_logistic_curve() -> None:
    assert expected_score(0.0) == 0.5
    assert expected_score(400.0) == pytest.approx(10.0 / 11.0)
    assert expected_score(-400.0) == pytest.approx(1.0 / 11.0)


def test_llr_is_zero_before_any_game() -> None:
    assert log_likelihood_ratio(Tally(), Sprt()) == 0.0


def test_llr_grows_with_wins_and_falls_with_losses() -> None:
    sprt = Sprt()
    even = log_likelihood_ratio(Tally(10, 0, 10), sprt)
    better = log_likelihood_ratio(Tally(15, 0, 5), sprt)
    worse = log_likelihood_ratio(Tally(5, 0, 15), sprt)
    assert worse < even < better


def test_a_few_wins_do_not_decide_anything() -> None:
    sprt = Sprt()
    assert verdict(log_likelihood_ratio(Tally(4, 0, 0), sprt), sprt) is None


def test_a_sweep_accepts_and_a_wipeout_rejects() -> None:
    sprt = Sprt()
    assert verdict(log_likelihood_ratio(Tally(40, 0, 0), sprt), sprt) == "accept"
    assert verdict(log_likelihood_ratio(Tally(0, 0, 40), sprt), sprt) == "reject"
    assert verdict(log_likelihood_ratio(Tally(10, 0, 10), sprt), sprt) is None


def test_node_limit_ignores_the_clock() -> None:
    limit = stockfish_limit(4_000, time_left_ms=120_000, increment_ms=500)
    assert limit.nodes == 4_000
    assert limit.white_clock is None and limit.black_clock is None


def test_clock_limit_mirrors_our_clock_to_both_sides() -> None:
    limit = stockfish_limit(None, time_left_ms=30_000, increment_ms=500)
    assert limit.nodes is None
    assert limit.white_clock == 30.0 and limit.black_clock == 30.0
    assert limit.white_inc == 0.5 and limit.black_inc == 0.5


def test_opponent_descriptions() -> None:
    assert describe_stockfish(2200, None) == "Stockfish UCI_Elo 2200"
    assert describe_stockfish(None, 4_000) == "Stockfish at 4000 nodes"
