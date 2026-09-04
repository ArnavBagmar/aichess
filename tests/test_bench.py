"""The bench's arithmetic and scheduling, tested without launching a single game."""

import pytest

from tools.elo_bench import Sprt, Tally, expected_score, log_likelihood_ratio, verdict


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
