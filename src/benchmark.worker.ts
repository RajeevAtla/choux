import { defaultDevice, init, jit, numpy as np } from '@jax-js/jax';
import {
  BATCH_SIZE,
  type BenchmarkRequest,
  type BenchmarkResult,
} from './benchmark';
import { forward, HIDDEN, INPUTS, WEIGHT_COUNT } from './model';

const score = jit(forward);
const worker: DedicatedWorkerGlobalScope =
  self as unknown as DedicatedWorkerGlobalScope;
let parameters: [np.Array, np.Array, np.Array, np.Array] | undefined;

function send(result: BenchmarkResult) {
  if (result.status === 'measured') {
    worker.postMessage(result, [result.scores.buffer]);
  } else {
    worker.postMessage(result);
  }
}

function makeBatch(offset: number) {
  const features = new Float32Array(BATCH_SIZE * INPUTS);
  for (let row = 0; row < BATCH_SIZE; row++) {
    for (let piece = 0; piece < 32; piece++) {
      const square = (piece * 17 + (offset + row) * 7) % 64;
      features[row * INPUTS + (piece % 12) * 64 + square] = 1;
    }
    features[row * INPUTS + INPUTS - 1] = (offset + row) % 2;
  }
  return features;
}

function referenceScore(
  features: Float32Array,
  row: number,
  weights: Float32Array,
) {
  const hiddenStart = INPUTS * HIDDEN;
  let result = weights[WEIGHT_COUNT - 1];
  for (let unit = 0; unit < HIDDEN; unit++) {
    let hidden = weights[hiddenStart + unit];
    for (let input = 0; input < INPUTS; input++) {
      hidden += features[row * INPUTS + input] * weights[input * HIDDEN + unit];
    }
    result += Math.max(0, hidden) * weights[hiddenStart + HIDDEN + unit];
  }
  return result;
}

async function infer(features: Float32Array) {
  if (!parameters) throw new Error('Load the model before running inference.');
  const scores = score(
    np.array(features, { shape: [BATCH_SIZE, INPUTS] }),
    parameters[0].ref,
    parameters[1].ref,
    parameters[2].ref,
    parameters[3].ref,
  );
  try {
    return Float32Array.from(await scores.data());
  } finally {
    scores.dispose();
  }
}

async function load({
  backend,
  checkpointUrl,
}: Extract<BenchmarkRequest, { backend: string }>) {
  const start = performance.now();
  if (!(await init(backend)).includes(backend)) {
    send({
      status: 'unavailable',
      message: `${backend} is unavailable in this browser.`,
    });
    return;
  }
  defaultDevice(backend);

  const response = await fetch(checkpointUrl);
  if (!response.ok)
    throw new Error(`Checkpoint request failed (${response.status}).`);
  const bytes = new DataView(await response.arrayBuffer());
  if (bytes.byteLength !== WEIGHT_COUNT * 4)
    throw new Error('Unexpected checkpoint size.');
  const weights = Float32Array.from({ length: WEIGHT_COUNT }, (_, i) =>
    bytes.getFloat32(i * 4, true),
  );
  const hiddenStart = INPUTS * HIDDEN;
  const outputStart = hiddenStart + HIDDEN;
  parameters = [
    np.array(weights.slice(0, hiddenStart), { shape: [INPUTS, HIDDEN] }),
    np.array(weights.slice(hiddenStart, outputStart)),
    np.array(weights.slice(outputStart, -1), { shape: [HIDDEN, 1] }),
    np.array(weights.slice(-1)),
  ];

  const setupMs = performance.now() - start;
  const warmStart = performance.now();
  const features = makeBatch(0);
  const scores = await infer(features);
  const firstBatchMs = performance.now() - warmStart;
  let maxError = 0;
  for (let row = 0; row < BATCH_SIZE; row++) {
    maxError = Math.max(
      maxError,
      Math.abs(scores[row] - referenceScore(features, row, weights)),
    );
  }
  if (!(maxError <= 0.00001))
    throw new Error(`Scores failed the reference check (${maxError}).`);
  send({ status: 'ready', setupMs, firstBatchMs, maxError });
}

worker.onmessage = async ({ data }: MessageEvent<BenchmarkRequest>) => {
  try {
    if ('backend' in data) {
      await load(data);
      return;
    }

    const scores = new Float32Array(data.workload);
    for (let offset = 0; offset < scores.length; offset += BATCH_SIZE) {
      scores.set(await infer(makeBatch(offset + data.trial)), offset);
    }
    send({ status: 'measured', scores });
  } catch (error) {
    send({
      status: 'error',
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
