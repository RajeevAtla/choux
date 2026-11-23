import {
  blockUntilReady,
  defaultDevice,
  grad,
  init,
  jit,
  numpy as np,
} from '@jax-js/jax';
import { references, tensors, unpack } from '../src/checkpoint';
import { forward, INPUTS, WEIGHT_COUNT } from '../src/model';
import { datasetPath, readGames } from './dataset';
import { type Example, fromGames, generate, random } from './examples';

type Weights = ReturnType<typeof tensors>;
const batchSize = 64;
const gradient = grad(
  (weights: Weights, features: np.Array, targets: np.Array) =>
    np.square(forward(features, ...weights).sub(targets)).mean(),
);
const step = jit(
  (weights: Weights, features: np.Array, targets: np.Array): Weights => {
    const update = gradient(references(weights), features, targets);
    return [
      weights[0].sub(update[0].mul(0.032)),
      weights[1].sub(update[1].mul(0.032)),
      weights[2].sub(update[2].mul(0.032)),
      weights[3].sub(update[3].mul(0.032)),
    ];
  },
);
const score = jit(forward);

function batch(examples: Example[], start: number) {
  const features = new Float32Array(batchSize * INPUTS);
  const values = new Float32Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    const example = examples[(start + i) % examples.length];
    features.set(example.features, i * INPUTS);
    values[i] = example.value;
  }
  return {
    features: np.array(features, { shape: [batchSize, INPUTS] }),
    values,
  };
}

async function error(weights: Weights, examples: Example[]) {
  let loss = 0;
  for (let offset = 0; offset < examples.length; offset += batchSize) {
    const { features, values } = batch(examples, offset);
    const predictions = await score(features, ...references(weights)).data();
    for (let i = 0; i < Math.min(batchSize, examples.length - offset); i++) {
      loss += Math.abs(Number(predictions[i]) - values[i]);
    }
  }
  return loss / examples.length;
}

async function serialize(weights: Weights) {
  const bytes = new ArrayBuffer(WEIGHT_COUNT * 4);
  const view = new DataView(bytes);
  let offset = 0;
  for (const tensor of references(weights)) {
    for (const value of await tensor.data()) {
      view.setFloat32(offset, Number(value), true);
      offset += 4;
    }
  }
  return bytes;
}

if (!(await init('wasm')).includes('wasm'))
  throw new Error('Wasm initialization failed.');
defaultDevice('wasm');
const [
  output = 'public/model.f32',
  dataset = datasetPath,
  incumbent = output,
  selfPlayDataset,
] = Bun.argv.slice(2);
const randomize = random(48902);
let weights = tensors(
  Float32Array.from({ length: WEIGHT_COUNT }, () => (randomize() - 0.5) * 0.16),
);
const validation = generate(16, 93285);
const baseline = await error(weights, validation);
const incumbentFile = Bun.file(incumbent);
const bestPublished = (await incumbentFile.exists())
  ? await error(tensors(unpack(await incumbentFile.arrayBuffer())), validation)
  : Infinity;
const games = (await readGames(dataset)).slice(-256);
const uniqueGames = new Map<string, { moves: string[] }>(
  games.map((game) => [game.moves.join(' '), game]),
);
if (selfPlayDataset) {
  const selfPlay = await readGames<{ moves: string[] }>(selfPlayDataset);
  if (!selfPlay.length)
    throw new Error(`No self-play games found at ${selfPlayDataset}.`);
  const existing = uniqueGames.size;
  for (const game of selfPlay) uniqueGames.set(game.moves.join(' '), game);
  console.log(`Added ${uniqueGames.size - existing} distinct self-play games.`);
}
const training = generate(144, 12459).concat(
  fromGames([...uniqueGames.values()]),
);
console.log(
  `Training on ${training.length} positions, validating on ${validation.length}. Random MAE ${baseline.toFixed(4)}.`,
);

let best = Infinity;
let checkpoint: ArrayBuffer | undefined;
for (let epoch = 0; epoch < 12; epoch++) {
  for (let i = training.length - 1; i > 0; i--) {
    const j = Math.floor(randomize() * (i + 1));
    [training[i], training[j]] = [training[j], training[i]];
  }
  for (let offset = 0; offset < training.length; offset += batchSize) {
    const { features, values } = batch(training, offset);
    weights = await blockUntilReady(
      step(weights, features, np.array(values, { shape: [batchSize, 1] })),
    );
  }
  const validationError = await error(weights, validation);
  console.log(
    `Epoch ${epoch + 1} validation MAE ${validationError.toFixed(4)}`,
  );
  if (validationError < best) {
    best = validationError;
    checkpoint = await serialize(weights);
  }
}

if (!checkpoint) throw new Error('Training produced no finite checkpoint.');
const roundTrip = await error(tensors(unpack(checkpoint)), validation);
if (Math.abs(roundTrip - best) > 1e-6)
  throw new Error('Exported checkpoint failed validation.');
const reportPath = Bun.env.CHOUX_TRAINING_REPORT;
const beatsRandom = best < baseline * 0.95;
const eligible = beatsRandom && best <= bestPublished + 0.000001;
if (!beatsRandom && !reportPath)
  throw new Error('Candidate missed the validation threshold.');
if (eligible || reportPath) {
  await Bun.write(output, checkpoint);
  console.log(`Saved ${output} with validation MAE ${roundTrip.toFixed(4)}.`);
}
if (!eligible) {
  console.log(
    `Candidate MAE ${best} did not pass the promotion gate. Retaining the published checkpoint.`,
  );
}
if (reportPath) {
  await Bun.write(
    reportPath,
    JSON.stringify({
      eligible,
      candidateMae: roundTrip,
      incumbentMae: Number.isFinite(bestPublished) ? bestPublished : null,
      trainingPositions: training.length,
    }),
  );
}

// The Wasm thread pool can keep Bun alive after the checkpoint has been written.
process.exit(0);
