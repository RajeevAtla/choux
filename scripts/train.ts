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
import {
  type Example,
  fromGames,
  generate,
  positionKey,
  random,
  type TrainingGame,
} from './examples';
import { referencePositions } from './reference-positions';
import {
  labelWithStockfish,
  stockfishValidationFens,
} from './stockfish-training';

type Weights = ReturnType<typeof tensors>;
const batchSize = 64;
const maxEpochs = 36;
const patience = 4;
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
const incumbentFile = Bun.file(incumbent);
const incumbentBytes = (await incumbentFile.exists())
  ? await incumbentFile.arrayBuffer()
  : undefined;
const games = (await readGames(dataset)).slice(-256);
const seen = new Set<string>();
const distinct = (game: { moves: string[] }) => {
  const key = game.moves.join(' ');
  if (seen.has(key)) return false;
  seen.add(key);
  return true;
};
const submittedGames = games.filter(distinct);
let selfPlayGames: TrainingGame[] = [];
if (selfPlayDataset) {
  const selfPlay = await readGames<TrainingGame>(selfPlayDataset);
  if (!selfPlay.length)
    throw new Error(`No self-play games found at ${selfPlayDataset}.`);
  selfPlayGames = selfPlay.filter(distinct);
}
const excludedPositions = new Set(
  referencePositions
    .map(({ fen }) => positionKey(fen))
    .concat(stockfishValidationFens.map(positionKey)),
);
const stockfishEnabled = Bun.env.CHOUX_TRAINING_STOCKFISH === '1';
if (stockfishEnabled && !incumbentBytes)
  throw new Error(
    'Targeted Stockfish labeling requires an incumbent checkpoint.',
  );
const stockfish =
  stockfishEnabled && incumbentBytes
    ? await labelWithStockfish(
        Bun.env.STOCKFISH ?? 'stockfish',
        incumbentBytes,
        submittedGames,
        selfPlayGames,
        excludedPositions,
      )
    : undefined;
if (stockfish && Bun.env.CHOUX_POLICY_POSITIONS)
  await Bun.write(
    Bun.env.CHOUX_POLICY_POSITIONS,
    JSON.stringify(stockfish.policyPositions),
  );
const heuristicValidation = generate(16, 93285);
const validation = stockfish?.validation ?? heuristicValidation;
const baseline = await error(weights, validation);
const incumbentWeights = incumbentBytes
  ? tensors(unpack(incumbentBytes))
  : undefined;
const bestPublished = incumbentWeights
  ? await error(incumbentWeights, validation)
  : Infinity;
const submittedPositions = fromGames(
  submittedGames,
  excludedPositions,
  stockfish?.labels,
);
const selfPlayPositions = fromGames(
  selfPlayGames,
  excludedPositions,
  stockfish?.labels,
);
const training = generate(144, 12459, excludedPositions).concat(
  submittedPositions,
  selfPlayPositions,
);
const labeled = training.filter(({ stockfish }) => stockfish);
const stockfishPositions = labeled.length;
// Keep Stockfish examples at about 15% of the training set when available.
const extraLabels = Math.max(
  0,
  Math.ceil((0.15 * training.length - stockfishPositions) / 0.85),
);
if (stockfishPositions)
  for (let i = 0; i < extraLabels; i++)
    training.push(labeled[i % stockfishPositions]);
const stockfishTrainingSamples =
  stockfishPositions + (stockfishPositions ? extraLabels : 0);
console.log(
  `Sampled ${submittedPositions.length} positions from ${submittedGames.length} issue games and ${selfPlayPositions.length} positions from ${selfPlayGames.length} self-play games.`,
);
console.log(
  `Used Stockfish scores for ${stockfishPositions} positions (${stockfishTrainingSamples} training samples).`,
);
if (stockfish)
  console.log(
    `Screened ${stockfish.screened} positions, retaining ${stockfish.mistakes} mistakes and ${stockfish.ordinary} ordinary positions.`,
  );
console.log(
  `Training on ${training.length} samples, validating on ${validation.length} ${stockfish ? 'Stockfish' : 'heuristic'} positions. Random MAE ${baseline.toFixed(4)}.`,
);

let best = Infinity;
let bestEpoch = 0;
let epochsRun = 0;
let checkpoint: ArrayBuffer | undefined;
for (let epoch = 1; epoch <= maxEpochs; epoch++) {
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
  epochsRun = epoch;
  console.log(`Epoch ${epoch} validation MAE ${validationError.toFixed(4)}`);
  if (validationError < best) {
    best = validationError;
    bestEpoch = epoch;
    checkpoint = await serialize(weights);
  }
  if (epoch - bestEpoch >= patience) {
    console.log(`Early stopping after ${patience} epochs without improvement.`);
    break;
  }
}

if (!checkpoint) throw new Error('Training produced no finite checkpoint.');
console.log(`Best validation MAE at epoch ${bestEpoch} of ${epochsRun}.`);
const candidateWeights = tensors(unpack(checkpoint));
const roundTrip = await error(candidateWeights, validation);
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
      epochsRun,
      bestEpoch,
      issueGames: submittedGames.length,
      issuePositions: submittedPositions.length,
      selfPlayGames: selfPlayGames.length,
      selfPlayPositions: selfPlayPositions.length,
      stockfishPositions,
      stockfishTrainingSamples,
      stockfishScreenedPositions: stockfish?.screened ?? 0,
      stockfishMistakes: stockfish?.mistakes ?? 0,
      stockfishOrdinaryPositions: stockfish?.ordinary ?? 0,
      validationSource: stockfish ? 'stockfish-v1' : 'heuristic',
      validationPositions: validation.length,
      heuristicCandidateMae: stockfish
        ? await error(candidateWeights, heuristicValidation)
        : roundTrip,
      heuristicIncumbentMae:
        stockfish && incumbentWeights
          ? await error(incumbentWeights, heuristicValidation)
          : Number.isFinite(bestPublished)
            ? bestPublished
            : null,
    }),
  );
}

// The Wasm thread pool can keep Bun alive after the checkpoint has been written.
process.exit(0);
