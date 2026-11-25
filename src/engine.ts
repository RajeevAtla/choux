import { defaultDevice, init, jit, numpy as np } from '@jax-js/jax';
import type { Chess, Move } from 'chess.js';
import { references, tensors, unpack } from './checkpoint';
import { forward, INPUTS } from './model';
import { readMovePolicy } from './move-policy';
import { encode } from './position';

const batchSize = 64;
export const moveBudgetMs = 200;
const score = jit(forward);
type Evaluate = ((
  features: Float32Array<ArrayBuffer>,
) => Promise<Float32Array>) & {
  movePrior?: ReturnType<typeof readMovePolicy>;
};
type RankedMove = {
  move: Move;
  score: number;
  terminal: boolean;
  order: number;
  children?: RankedMove[];
};

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
  return Object.assign(evaluate, { movePrior: readMovePolicy(checkpoint) });
}

async function scoreMoves(
  game: Chess,
  evaluate: Evaluate,
  legal: Move[],
  usePolicy: boolean,
  deadline = Infinity,
) {
  const ranked: RankedMove[] = [];
  const prior = usePolicy ? evaluate.movePrior : undefined;
  for (let start = 0; start < legal.length; start += batchSize) {
    const candidates = legal.slice(start, start + batchSize);
    const inputs = new Float32Array(batchSize * INPUTS);
    const terminal: (number | undefined)[] = [];
    for (const [index, move] of candidates.entries()) {
      if (performance.now() >= deadline) return undefined;
      game.move(move);
      let value: number | undefined;
      if (game.isCheckmate()) value = 2;
      else if (game.isDraw()) value = 0;
      else encode(game, inputs, index * INPUTS);
      game.undo();
      terminal.push(value);
    }
    if (performance.now() >= deadline) return undefined;
    const values = await evaluate(inputs);
    for (const [index, move] of candidates.entries()) {
      const exact = terminal[index];
      const value = exact ?? -Math.max(-1, Math.min(1, values[index]));
      ranked.push({
        move,
        score: value,
        terminal: exact !== undefined,
        order: value + (prior ? 0.15 * Math.tanh(prior(game, move)) : 0),
      });
    }
  }
  return ranked.sort((a, b) => b.order - a.order);
}

export async function rankMoves(game: Chess, evaluate: Evaluate) {
  return (
    (await scoreMoves(game, evaluate, game.moves({ verbose: true }), false)) ??
    []
  );
}

async function chooseMoveShallow(
  game: Chess,
  evaluate: Evaluate,
  budgetMs: number,
  usePolicy: boolean,
) {
  if (game.isGameOver()) return undefined;
  const deadline = performance.now() + budgetMs;
  const legal = game.moves({ verbose: true });
  const ranked = (await scoreMoves(game, evaluate, legal, usePolicy)) ?? [];
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

export function chooseMoveBaseline(
  game: Chess,
  evaluate: Evaluate,
  budgetMs = moveBudgetMs,
) {
  return chooseMoveShallow(game, evaluate, budgetMs, false);
}

export function chooseMove(
  game: Chess,
  evaluate: Evaluate,
  budgetMs = moveBudgetMs,
) {
  return chooseMoveShallow(game, evaluate, budgetMs, true);
}

function forcing(move: Move) {
  return move.captured || move.promotion || /[+#]$/.test(move.san);
}

export async function chooseMoveTactical(
  game: Chess,
  evaluate: Evaluate,
  budgetMs = moveBudgetMs,
) {
  if (game.isGameOver()) return undefined;
  const deadline = performance.now() + budgetMs;
  const legal = game.moves({ verbose: true });
  const mate = legal.find((move) => move.san.endsWith('#'));
  if (mate || legal.length <= 1) return mate ?? legal[0];
  const ranked = await scoreMoves(game, evaluate, legal, true, deadline);
  if (!ranked?.length) return legal[0];
  const first = ranked[0];

  const root = ranked.slice(0, 8);
  const expired = () => performance.now() >= deadline;

  async function search(
    previous: RankedMove,
    depth: number,
    extensions: number,
    alpha: number,
    beta: number,
  ): Promise<number | undefined> {
    const estimate = -previous.score;
    if (previous.terminal || (depth === 0 && extensions === 0)) return estimate;
    if (expired()) return undefined;

    game.move(previous.move);
    try {
      const inCheck = game.isCheck();
      let value = depth === 0 && !inCheck ? estimate : -Infinity;
      if (value >= beta) return value;
      alpha = Math.max(alpha, value);

      let children: RankedMove[] | undefined;
      if (depth === 0 && !inCheck) {
        if (previous.children) {
          children = previous.children.filter(({ move }) => forcing(move));
        } else {
          const captures = game.moves({ verbose: true }).filter(forcing);
          children = await scoreMoves(game, evaluate, captures, true, deadline);
        }
      } else {
        previous.children ??= await scoreMoves(
          game,
          evaluate,
          game.moves({ verbose: true }),
          true,
          deadline,
        );
        children = previous.children;
      }
      if (!children || expired()) return undefined;
      // At one ply remaining the model scores all replies in one batch.
      if (depth === 1 && extensions === 0)
        return Math.max(...children.map(({ score }) => score));
      const selected = inCheck ? children : children.slice(0, 4);
      for (const child of selected) {
        const childValue = await search(
          child,
          Math.max(0, depth - 1),
          extensions - Number(depth === 0),
          -beta,
          -alpha,
        );
        if (childValue === undefined) return undefined;
        value = Math.max(value, -childValue);
        if (value >= beta) break;
        alpha = Math.max(alpha, value);
      }
      return value;
    } finally {
      game.undo();
    }
  }

  let best = first;
  for (const [depth, extensions] of [
    [1, 0],
    [1, 2],
    [2, 2],
  ]) {
    let bestScore = -Infinity;
    let next = best;
    const ordered = [best, ...root.filter((candidate) => candidate !== best)];
    for (const candidate of ordered) {
      if (expired()) return best.move;
      const response = await search(
        candidate,
        depth,
        extensions,
        -Infinity,
        -bestScore,
      );
      if (response === undefined) return best.move;
      const value = -response;
      if (value > bestScore) {
        next = candidate;
        bestScore = value;
      }
      // Keep the best completed two-ply result if its full root cannot finish.
      if (depth === 1 && extensions === 0) best = next;
    }
    best = next;
    if (bestScore === 2) break;
  }
  return best.move;
}
