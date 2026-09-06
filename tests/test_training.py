import random
import unittest
from collections import defaultdict

import chess
import numpy as np

from tools.label_positions import random_position
from tools.train_evaluator import FEATURES, features
from tools.train_move_policy import POLICY_FEATURES, move_features
from tools.train_nnue import FEATURES as NNUE_FEATURES
from tools.train_nnue import Example, halfkp_indices, metrics
from tools.train_ranked_nnue import game_target, sigmoid


class TrainingPipelineTests(unittest.TestCase):
    def test_ranked_training_targets_follow_side_to_move(self) -> None:
        self.assertEqual(game_target("1-0", chess.WHITE), 1.0)
        self.assertEqual(game_target("1-0", chess.BLACK), 0.0)
        self.assertEqual(game_target("0-1", chess.BLACK), 1.0)
        self.assertEqual(game_target("1/2-1/2", chess.WHITE), 0.5)
        self.assertIsNone(game_target("*", chess.WHITE))
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_nnue_metrics_measure_the_deployed_blend(self) -> None:
        example = Example(
            us=np.empty(0, dtype=np.int32),
            them=np.empty(0, dtype=np.int32),
            target=2.0,
            teacher_residual=1.0,
            baseline=0.0,
            validation=True,
            phase=0,
            phase_bucket=0,
        )
        embedding = np.empty((0, 1), dtype=np.float32)
        hidden_bias = np.zeros(1, dtype=np.float32)
        output = np.zeros(2, dtype=np.float32)
        mae, rmse, baseline_mae = metrics(
            embedding,
            hidden_bias,
            output,
            np.float32(2.0),
            [example],
            deployment_blend=0.5,
        )
        self.assertEqual((mae, rmse, baseline_mae), (0.0, 0.0, 400.0))

    def test_halfkp_features_are_bounded_and_perspective_sensitive(self) -> None:
        board = chess.Board()
        white = halfkp_indices(board, chess.WHITE)
        black = halfkp_indices(board, chess.BLACK)

        self.assertEqual(len(white), 30)
        self.assertEqual(len(black), 30)
        self.assertTrue(all(0 <= int(index) < NNUE_FEATURES for index in white))
        self.assertFalse((white == black).all())

    def test_feature_indices_are_bounded_and_mirror_antisymmetric(self) -> None:
        board = chess.Board("r3k2r/pp2bppp/2n1pn2/2bp4/8/2N1PN2/PPQB1PPP/2RR2K1 w kq - 4 13")
        vector: defaultdict[int, float] = defaultdict(float)
        mirrored: defaultdict[int, float] = defaultdict(float)
        for index, value in features(board):
            vector[index] += value
        for index, value in features(board.mirror()):
            mirrored[index] += value
        self.assertTrue(all(0 <= index < FEATURES for index in vector))
        self.assertEqual(vector.keys(), mirrored.keys())
        for index, value in vector.items():
            self.assertAlmostEqual(value, -mirrored[index])

    def test_seeded_position_sampling_is_legal_and_reproducible(self) -> None:
        first = random_position(random.Random(7), 12, 12)
        second = random_position(random.Random(7), 12, 12)
        self.assertEqual(first.fen(), second.fen())
        self.assertTrue(first.is_valid())
        self.assertFalse(first.is_game_over())

    def test_move_policy_features_cover_all_legal_move_types(self) -> None:
        board = chess.Board("4k3/P6p/8/3pP3/8/8/7P/4K3 w - d6 0 1")
        moves = list(board.legal_moves)
        encoded = [move_features(board, move) for move in moves]
        self.assertTrue(encoded)
        self.assertTrue(all(len(features) == 12 for features in encoded))
        self.assertTrue(
            all(0 <= index < POLICY_FEATURES for features in encoded for index in features)
        )
        self.assertEqual(encoded, [move_features(board, move) for move in moves])


if __name__ == "__main__":
    unittest.main()
