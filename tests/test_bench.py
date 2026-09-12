"""The bench's arithmetic and scheduling, tested without launching a single game."""

from pathlib import Path

import pytest

from tools.elo_bench import (
    Sprt,
    Tally,
    check_agent_dir,
    describe_stockfish,
    expected_score,
    log_likelihood_ratio,
    run_pairs,
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


def test_run_pairs_stops_scheduling_after_the_verdict() -> None:
    played: list[str] = []
    seen: list[str] = []

    def play(fen: str) -> str:
        played.append(fen)
        return fen

    def on_pair(result: str) -> bool:
        seen.append(result)
        return len(seen) >= 3

    run_pairs([str(i) for i in range(20)], workers=2, play=play, on_pair=on_pair)
    assert len(seen) >= 3
    # Only pairs already in flight when the verdict landed may finish after it.
    assert len(played) <= 3 + 2 - 1
    assert set(seen) == set(played)


def test_run_pairs_plays_everything_when_nothing_stops_it() -> None:
    seen: list[str] = []

    def on_pair(result: str) -> bool:
        seen.append(result)
        return False

    run_pairs(["a", "b", "c"], workers=2, play=lambda fen: fen, on_pair=on_pair)
    assert sorted(seen) == ["a", "b", "c"]


def test_agent_dir_must_hold_agent_and_weights(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"agent.py"):
        check_agent_dir(tmp_path)
    (tmp_path / "agent.py").write_text("")
    with pytest.raises(SystemExit, match=r"nnue.npz"):
        check_agent_dir(tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "nnue.npz").write_bytes(b"")
    assert check_agent_dir(tmp_path) == tmp_path.resolve()
