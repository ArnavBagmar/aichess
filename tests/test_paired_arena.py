from __future__ import annotations

import unittest

from tools.paired_arena import score_interval, score_to_elo


class PairedArenaStatisticsTests(unittest.TestCase):
    def test_confirmation_interval_uses_pair_variance(self) -> None:
        pair_scores = [0.0] * 10 + [0.5] * 8 + [1.0] * 20 + [1.5] * 11 + [2.0]

        lower, upper = score_interval(pair_scores)

        self.assertAlmostEqual(lower, 0.3493, places=4)
        self.assertAlmostEqual(upper, 0.5007, places=4)

    def test_elo_conversion_is_symmetric(self) -> None:
        self.assertAlmostEqual(score_to_elo(0.5), 0.0)
        self.assertAlmostEqual(score_to_elo(0.4), -score_to_elo(0.6))

    def test_single_pair_reports_uninformative_interval(self) -> None:
        self.assertEqual(score_interval([1.0]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
