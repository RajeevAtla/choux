import type { Chess, Move, Square } from 'chess.js';
import { POLICY_WEIGHTS, WEIGHT_COUNT } from './modelShape';

const pieces = 'pnbrqk';
const squareFeatures = 6 * 64 * 2;

export function policyFeatures(game: Chess, move: Move) {
  const verticalFlip = game.turn() === 'b' ? 56 : 0;
  function index(square: Square) {
    return (
      ((square.charCodeAt(1) - 49) * 8 + square.charCodeAt(0) - 97) ^
      verticalFlip
    );
  }
  const from = index(move.from);
  const to = index(move.to);
  const mover = pieces.indexOf(move.piece);
  const features = [mover * 128 + from, mover * 128 + 64 + to];
  if (move.captured)
    features.push(squareFeatures + pieces.indexOf(move.captured));
  if (move.promotion)
    features.push(squareFeatures + 6 + pieces.indexOf(move.promotion));
  if (move.san.includes('+')) features.push(squareFeatures + 12);
  if (move.san.includes('#')) features.push(squareFeatures + 13);
  if (move.san.startsWith('O-O')) features.push(squareFeatures + 14);
  if (game.isCheck()) features.push(squareFeatures + 15);
  if (from % 8 === to % 8) features.push(squareFeatures + 16);
  return features;
}

export function policyLogit(
  weights: Float32Array,
  features: readonly number[],
) {
  return features.reduce((score, feature) => score + weights[feature], 0);
}

export function readMovePolicy(checkpoint: ArrayBuffer) {
  if (checkpoint.byteLength !== (WEIGHT_COUNT + POLICY_WEIGHTS) * 4)
    return undefined;
  const view = new DataView(checkpoint, WEIGHT_COUNT * 4);
  const weights = Float32Array.from({ length: POLICY_WEIGHTS }, (_, i) =>
    view.getFloat32(i * 4, true),
  );
  if (!weights.every(Number.isFinite))
    throw new Error('Checkpoint contains invalid policy weights.');
  return (game: Chess, move: Move) =>
    policyLogit(weights, policyFeatures(game, move));
}
