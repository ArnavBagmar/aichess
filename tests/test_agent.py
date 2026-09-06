import time
import unittest

import chess

import agent
from agent import MATE, Engine, evaluate, get_move, static_exchange
from tools.benchmark import POSITIONS


class EvaluationTests(unittest.TestCase):
    def test_starting_position_is_symmetric(self) -> None:
        self.assertEqual(evaluate(chess.Board(), use_residual=False), 0)

    def test_mirror_preserves_side_to_move_score(self) -> None:
        board = chess.Board("4k3/7p/8/3P4/8/8/P7/4K3 w - - 0 1")
        self.assertEqual(evaluate(board), evaluate(board.mirror()))

    def test_extra_white_queen_has_correct_sign(self) -> None:
        white = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
        black = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
        self.assertGreater(evaluate(white), 800)
        self.assertLess(evaluate(black), -800)

    def test_cached_pawn_term_preserves_evaluation(self) -> None:
        engine = Engine()
        for position in POSITIONS:
            board = chess.Board(position.fen)
            self.assertEqual(engine._evaluate(board), evaluate(board), position.name)
        self.assertGreater(engine.stats.pawn_calls, 0)


class SearchTests(unittest.TestCase):
    def test_static_exchange_values_hanging_and_poisoned_captures(self) -> None:
        hanging = chess.Board("6k1/8/8/4q3/8/8/4R3/6K1 w - - 0 1")
        poisoned = chess.Board("4r1k1/8/8/4p3/8/8/4Q3/6K1 w - - 0 1")
        self.assertGreater(static_exchange(hanging, chess.Move.from_uci("e2e5")), 800)
        self.assertLess(static_exchange(poisoned, chess.Move.from_uci("e2e5")), -700)

    def test_mate_in_one(self) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        move = chess.Move.from_uci(get_move(board.fen(), 1_000))
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_wins_hanging_queen(self) -> None:
        board = chess.Board("6k1/8/8/4q3/8/8/4R3/6K1 w - - 0 1")
        self.assertEqual(get_move(board.fen(), 1_000), "e2e5")

    def test_captures_newly_promoted_queen_in_public_regression(self) -> None:
        board = chess.Board("2Q5/pb6/6p1/5p1p/7P/6Pk/5K2/8 b - - 0 48")
        self.assertEqual(get_move(board.fen(), 250), "b7c8")

    def test_low_clock_always_returns_legal_move_quickly(self) -> None:
        board = chess.Board()
        start = time.perf_counter()
        move = chess.Move.from_uci(get_move(board.fen(), 1))
        elapsed = time.perf_counter() - start
        self.assertIn(move, board.legal_moves)
        self.assertLess(elapsed, 0.1)

    def test_emergency_budget_keeps_a_hard_clock_reserve(self) -> None:
        for clock_ms in (100, 500, 1_000, 2_500, 5_000, 120_000):
            soft, hard = Engine._time_budget(clock_ms)
            self.assertGreater(soft, 0)
            self.assertGreaterEqual(hard, soft)
            self.assertLess(hard, clock_ms / 1_000)

    def test_terminal_position_is_rejected(self) -> None:
        checkmate = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
        with self.assertRaises(ValueError):
            get_move(checkmate, 1_000)

    def test_draw_detection_preserves_insufficient_material(self) -> None:
        self.assertTrue(Engine._is_draw(chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")))
        self.assertFalse(Engine._is_draw(chess.Board("7k/7p/8/8/8/8/8/K7 w - - 0 1")))

    def test_mate_scores_round_trip_through_tt_encoding(self) -> None:
        engine = Engine()
        for score in (MATE - 7, -MATE + 9, 123, -456):
            stored = engine._score_to_tt(score, 5)
            self.assertEqual(engine._score_from_tt(stored, 5), score)

    def test_search_exposes_completed_iteration_telemetry(self) -> None:
        engine = Engine()
        board = chess.Board()
        move = engine.choose(board, 1_000)
        self.assertIn(move, board.legal_moves)
        self.assertGreaterEqual(engine.stats.completed_depth, 1)
        self.assertGreater(engine.stats.nodes, 0)
        self.assertGreater(engine.stats.elapsed_s, 0)

    def test_pondering_stops_cleanly_before_the_next_move(self) -> None:
        board = chess.Board()
        first = chess.Move.from_uci(get_move(board.fen(), 1_000))
        board.push(first)
        board.push(next(iter(board.legal_moves)))

        second = chess.Move.from_uci(get_move(board.fen(), 100))

        self.assertIn(second, board.legal_moves)
        agent._stop_pondering()
        self.assertIsNone(agent._PONDER_THREAD)

    def test_pondering_is_disabled_near_flagfall(self) -> None:
        get_move(chess.Board().fen(), 100)

        self.assertIsNone(agent._PONDER_THREAD)

    def test_repeated_root_move_is_only_avoided_when_ahead(self) -> None:
        engine = Engine()
        ahead = chess.Board("6k1/8/8/8/8/8/R7/6K1 w - - 0 1")
        key = agent._tt_key(ahead)
        remembered = chess.Move.from_uci("a2a3")
        engine.root_choices[key] = remembered

        self.assertEqual(engine._repetition_move(ahead, key), remembered)

        equal = chess.Board()
        equal_key = agent._tt_key(equal)
        engine.root_choices[equal_key] = chess.Move.from_uci("e2e4")
        self.assertIsNone(engine._repetition_move(equal, equal_key))

    def test_benchmark_corpus_contains_legal_nonterminal_positions(self) -> None:
        phases = set()
        for position in POSITIONS:
            board = chess.Board(position.fen)
            self.assertTrue(board.is_valid(), position.name)
            self.assertFalse(board.is_game_over(), position.name)
            phases.add(position.phase)
        self.assertTrue({"opening", "middlegame", "tactical", "endgame"} <= phases)


if __name__ == "__main__":
    unittest.main()
