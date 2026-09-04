import random
import unittest
from collections import defaultdict

import chess

from tools.label_positions import random_position
from tools.train_evaluator import FEATURES, features
from tools.train_nnue import FEATURES as NNUE_FEATURES
from tools.train_nnue import halfkp_indices


class TrainingPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
