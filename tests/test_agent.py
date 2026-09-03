import time
import unittest

import chess

from agent import MATE, Engine, evaluate, get_move


class EvaluationTests(unittest.TestCase):
    def test_starting_position_is_symmetric(self) -> None:
        self.assertEqual(evaluate(chess.Board()), 0)

    def test_mirror_preserves_side_to_move_score(self) -> None:
        board = chess.Board("4k3/7p/8/3P4/8/8/P7/4K3 w - - 0 1")
        self.assertEqual(evaluate(board), evaluate(board.mirror()))

    def test_extra_white_queen_has_correct_sign(self) -> None:
        white = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
        black = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
        self.assertGreater(evaluate(white), 800)
        self.assertLess(evaluate(black), -800)


class SearchTests(unittest.TestCase):
    def test_mate_in_one(self) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci(get_move(board.fen(), 1_000))
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_wins_hanging_queen(self) -> None:
        board = chess.Board("6k1/8/8/4q3/8/8/4R3/6K1 w - - 0 1")
        self.assertEqual(get_move(board.fen(), 1_000), "e2e5")

    def test_low_clock_always_returns_legal_move_quickly(self) -> None:
        board = chess.Board()
        start = time.perf_counter()
        move = chess.Move.from_uci(get_move(board.fen(), 1))
        elapsed = time.perf_counter() - start
        self.assertIn(move, board.legal_moves)
        self.assertLess(elapsed, 0.1)

    def test_terminal_position_is_rejected(self) -> None:
        checkmate = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
        with self.assertRaises(ValueError):
            get_move(checkmate, 1_000)

    def test_mate_scores_round_trip_through_tt_encoding(self) -> None:
        engine = Engine()
        for score in (MATE - 7, -MATE + 9, 123, -456):
            stored = engine._score_to_tt(score, 5)
            self.assertEqual(engine._score_from_tt(stored, 5), score)


if __name__ == "__main__":
    unittest.main()
