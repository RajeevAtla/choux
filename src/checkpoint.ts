import { numpy as np } from '@jax-js/jax';
import { HIDDEN, INPUTS, WEIGHT_COUNT } from './model';
import { POLICY_WEIGHTS } from './modelShape';

export function unpack(bytes: ArrayBuffer) {
  if (
    bytes.byteLength !== WEIGHT_COUNT * 4 &&
    bytes.byteLength !== (WEIGHT_COUNT + POLICY_WEIGHTS) * 4
  )
    throw new Error('Unexpected checkpoint size.');
  const view = new DataView(bytes);
  const weights = Float32Array.from({ length: WEIGHT_COUNT }, (_, i) =>
    view.getFloat32(i * 4, true),
  );
  if (!weights.every(Number.isFinite))
    throw new Error('Checkpoint contains invalid weights.');
  return weights;
}

export function tensors(
  weights: Float32Array,
): [np.Array, np.Array, np.Array, np.Array] {
  const hiddenStart = INPUTS * HIDDEN;
  const outputStart = hiddenStart + HIDDEN;
  return [
    np.array(weights.slice(0, hiddenStart), { shape: [INPUTS, HIDDEN] }),
    np.array(weights.slice(hiddenStart, outputStart)),
    np.array(weights.slice(outputStart, -1), { shape: [HIDDEN, 1] }),
    np.array(weights.slice(-1)),
  ];
}

export function references(
  weights: ReturnType<typeof tensors>,
): ReturnType<typeof tensors> {
  return [weights[0].ref, weights[1].ref, weights[2].ref, weights[3].ref];
}
