import { nn, numpy as np } from '@jax-js/jax';

export { HIDDEN, INPUTS, WEIGHT_COUNT } from './modelShape';

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
