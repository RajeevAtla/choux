import { nn, numpy as np } from '@jax-js/jax';

export const INPUTS = 782;
export const HIDDEN = 64;
export const WEIGHT_COUNT = INPUTS * HIDDEN + HIDDEN * 2 + 1;

export function forward(
  positions: np.Array,
  inputWeights: np.Array,
  hiddenBias: np.Array,
  outputWeights: np.Array,
  outputBias: np.Array,
) {
  const hidden = nn.relu(np.matmul(positions, inputWeights).add(hiddenBias));
  return np.matmul(hidden, outputWeights).add(outputBias);
}
