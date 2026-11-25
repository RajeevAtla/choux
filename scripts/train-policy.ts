import { Chess } from 'chess.js';
import { evaluator, rankMoves } from '../src/engine';
import { POLICY_WEIGHTS, WEIGHT_COUNT } from '../src/modelShape';
import {
  policyFeatures,
  policyLogit,
  readMovePolicy,
} from '../src/move-policy';
import { coordinate } from '../src/position';
import { startStockfish } from './stockfish';

type Position = { fen: string; bestMove: string };
type Target = { features: number[]; weight: number };
const [modelPath, positionsPath] = Bun.argv.slice(2);
if (!modelPath || !positionsPath)
  throw new Error('Usage: bun scripts/train-policy.ts <candidate> <positions>');
const positions: Position[] = await Bun.file(positionsPath).json();
if (!positions.length) {
  console.log('No Stockfish positions available for policy training.');
  process.exit(0);
}

const checkpoint = await Bun.file(modelPath).arrayBuffer();
const evaluate = await evaluator(checkpoint);
const examples: Target[][] = [];
const stockfish = await startStockfish(Bun.env.STOCKFISH ?? 'stockfish');
try {
  const unique = new Map(positions.map((position) => [position.fen, position]));
  for (const { fen, bestMove } of [...unique.values()].slice(0, 64)) {
    const game = new Chess(fen);
    const legal = game.moves({ verbose: true });
    const best = legal.find((move) => coordinate(move) === bestMove);
    if (!best || legal.length < 2) continue;
    const ranked = await rankMoves(game, evaluate);
    const shortlist = [
      best,
      ...ranked
        .map(({ move }) => move)
        .filter((move) => coordinate(move) !== bestMove)
        .slice(0, 4),
    ];
    const targets: Target[] = [];
    for (const move of shortlist) {
      const { points } = await stockfish.analyze(fen, coordinate(move), {
        nodes: 4_000,
        moves: [],
      });
      targets.push({ features: policyFeatures(game, move), weight: points });
    }
    const strongest = Math.max(...targets.map(({ weight }) => weight));
    examples.push(
      targets.map(({ features, weight }) => ({
        features,
        weight: Math.exp((weight - strongest) * 8),
      })),
    );
  }
} finally {
  stockfish.close();
}
if (!examples.length) {
  console.log('No legal move shortlists available for policy training.');
  process.exit(0);
}

const weights = new Float32Array(POLICY_WEIGHTS);
for (let epoch = 0; epoch < 14; epoch++) {
  for (const targets of examples) {
    const logits = targets.map(({ features }) =>
      policyLogit(weights, features),
    );
    const max = Math.max(...logits);
    const predictions = logits.map((value) => Math.exp(value - max));
    const total = predictions.reduce((sum, value) => sum + value, 0);
    const targetTotal = targets.reduce((sum, { weight }) => sum + weight, 0);
    for (const [index, { weight, features }] of targets.entries()) {
      const update = 0.08 * (weight / targetTotal - predictions[index] / total);
      for (const feature of features) weights[feature] += update;
    }
  }
}

const bytes = new ArrayBuffer((WEIGHT_COUNT + POLICY_WEIGHTS) * 4);
new Uint8Array(bytes).set(new Uint8Array(checkpoint, 0, WEIGHT_COUNT * 4));
const view = new DataView(bytes, WEIGHT_COUNT * 4);
for (const [index, weight] of weights.entries())
  view.setFloat32(index * 4, weight, true);
if (!readMovePolicy(bytes))
  throw new Error('Policy checkpoint failed to load.');
await Bun.write(modelPath, bytes);
console.log(
  `Trained move policy on ${examples.length} Stockfish move shortlists.`,
);
process.exit(0);
