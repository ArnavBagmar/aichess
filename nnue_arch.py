"""Architecture and quantization constants for our NNUE.

The network is the nnue-pytorch master architecture restricted to the HalfKAv2_hm feature
set, sized down to L1=256. Everything here is a faithful transcription of the trainer's
defaults (model/quantize.py and model/modules/config.py in official-stockfish/nnue-pytorch);
the exporter in tools/ and the inference in nnue_engine.py must agree on these numbers, so
they live in one place.

Integer pipeline, derived from the trainer's fake-quantization with its default config
(both correction factors are exactly 1.0 there):

  accumulator   int16, FT weight rows summed per active feature, in 1/256 units
  ft activation clamp(acc, 0, 255), pairwise product of accumulator halves, >> 9  -> 0..127
  l1            int8 weights (1/128) x act (1/128) + int32 bias (1/16384) -> 1/16384 units
  l1 activation squared path (x*x) >> 21, linear path x >> 7, both clipped to 0..127
  l2            int8 weights (1/64) x act (1/128) + int32 bias (1/8192)  -> 1/8192 units
  l2 activation squared path (x*x) >> 19, linear path x >> 6, both clipped to 0..127
  output        int8 weights (1/128) x act (1/128) + int32 bias (1/16384) -> 1/16384 units
  skip          l1c[L2-2] - l1c[L2-1], added to the output in 1/16384 units
  final         trunc(x * 9600 / 16384) -> 1/9600 units, then +- (wpsqt - bpsqt) / 2
"""

from typing import Final

# Feature set: HalfKAv2_hm with the 12-plane training layout (no king merging).
NUM_SQ: Final = 64
NUM_PT: Final = 12
NUM_PLANES: Final = NUM_SQ * NUM_PT  # 768
NUM_KING_BUCKETS: Final = 32
NUM_FEATURES: Final = NUM_PLANES * NUM_KING_BUCKETS  # 24,576
MAX_ACTIVE_FEATURES: Final = 32

# Network dimensions (trainer flags: --l1 256, defaults for the rest).
L1: Final = 256
L2: Final = 32
L3: Final = 32
NUM_LS_BUCKETS: Final = 8
NUM_PSQT_BUCKETS: Final = 8
OUT_IN: Final = 2 * L2 + 2 * L3  # 128, input width of the output layer

# Quantization scales (trainer QuantizationConfig defaults).
NNUE2SCORE: Final = 600.0
FT_QUANT_ONE: Final = 256
FT_ACT_MAX: Final = 255
HIDDEN_QUANT_ONE: Final = 128
HIDDEN_ACT_MAX: Final = 127
WEIGHT_SCALE_L1: Final = 128
WEIGHT_SCALE_L2: Final = 64
WEIGHT_SCALE_OUT: Final = 128
PSQT_WEIGHT_SCALE: Final = 9600  # nnue2score * weight_scale_out(=16)

# Integer shift amounts implied by the scales above.
FT_PAIRWISE_SHIFT: Final = 9  # (a*b) in 1/65536 -> 1/128 units
L1_LINEAR_SHIFT: Final = 7  # 1/16384 -> 1/128
L1_SQUARE_SHIFT: Final = 21  # (1/16384)^2 -> 1/128
L2_LINEAR_SHIFT: Final = 6  # 1/8192 -> 1/128
L2_SQUARE_SHIFT: Final = 19  # (1/8192)^2 -> 1/128
OUTPUT_MUL: Final = 9600  # nnue2score * weight_scale_out(=16.0)
OUTPUT_DIV: Final = 16384  # hidden_quant_one * weight_scale_l_out(=128), see docstring

# A final score is (2*output + sign*(wpsqt - bpsqt)) in 1/19200-of-nnue2score units,
# which makes one centipawn exactly SCORE_PER_CP internal units.
SCORE_PER_CP: Final = 32

WEIGHTS_FILE: Final = "nnue.npz"
