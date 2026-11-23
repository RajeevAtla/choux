import { defaultDevice, init, jit, numpy as np } from '@jax-js/jax';
import type { Chess, Move } from 'chess.js';
import { references, tensors, unpack } from './checkpoint';
import { forward, INPUTS } from './model';
import { encode } from './position';

const batchSize = 64;
const score = jit(forward);
type Evaluate = Awaited<ReturnType<typeof evaluator>>;

export async function evaluator(checkpoint: ArrayBuffer) {
  if (!(await init('wasm')).includes('wasm'))
    throw new Error('This browser cannot start the Wasm engine.');
  defaultDevice('wasm');
  const weights = tensors(unpack(checkpoint));
  async function evaluate(features: Float32Array<ArrayBuffer>) {
    const inputs = np.array(features, { shape: [batchSize, INPUTS] });
    return Float32Array.from(
      await score(inputs, ...references(weights)).data(),
    );
  }
  await evaluate(new Float32Array(batchSize * INPUTS));
  return evaluate;
}

export async function rankMoves(game: Chess, evaluate: Evaluate) {
  const legal = game.moves({ verbose: true });
  const ranked: { move: Move; score: number }[] = [];
  for (let start = 0; start < legal.length; start += batchSize) {
    const candidates = legal.slice(start, start + batchSize);
    const inputs = new Float32Array(batchSize * INPUTS);
    const terminal = candidates.map((move, index) => {
      game.move(move);
      let value: number | undefined;
      if (game.isCheckmate()) value = 2;
      else if (game.isDraw()) value = 0;
      else encode(game, inputs, index * INPUTS);
      game.undo();
      return value;
    });
    const values = await evaluate(inputs);
    ranked.push(
      ...candidates.map((move, index) => ({
        move,
        score: terminal[index] ?? -Math.max(-1, Math.min(1, values[index])),
      })),
    );
  }
  return ranked.sort((a, b) => b.score - a.score);
}

export async function chooseMove(
  game: Chess,
  evaluate: Evaluate,
  budgetMs = 200,
) {
  if (game.isGameOver()) return undefined;
  const deadline = performance.now() + budgetMs;
  const ranked = await rankMoves(game, evaluate);
  const first = ranked[0];
  if (first.score === 2) return first.move;

  let best: { move: Move; score: number } | undefined;
  for (const { move } of ranked.slice(0, 8)) {
    if (performance.now() >= deadline) break;
    game.move(move);
    const reply = game.isDraw()
      ? undefined
      : (await rankMoves(game, evaluate))[0];
    game.undo();
    const score = reply ? -reply.score : 0;
    if (!best || best.score < score) best = { move, score };
  }
  return best?.move ?? first.move;
}
