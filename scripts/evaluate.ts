import { Chess, type Color } from 'chess.js';
import { chooseMove, evaluator } from '../src/engine';
import { INPUTS } from '../src/model';
import { replay, result } from '../src/position';
import { parseSubmission } from './dataset';
import { random } from './examples';
import { material } from './teacher';

const [checkpoint = 'public/model.f32', mode] = Bun.argv.slice(2);
const evaluate = await evaluator(await Bun.file(checkpoint).arrayBuffer());
const scores = await evaluate(new Float32Array(64 * INPUTS));
if (scores.length !== 64 || !scores.every(Number.isFinite))
  throw new Error('Invalid model scores.');

const mate = new Chess('7k/8/5KQ1/8/8/8/8/8 w - - 0 1');
const finish = await chooseMove(mate, evaluate);
if (!finish?.san.endsWith('#')) throw new Error('Engine missed mate in one.');

const openings = [
  ['e2e4', 'e7e5', 'g1f3', 'b8c6'],
  ['d2d4', 'd7d5', 'c2c4', 'e7e6'],
];
for (const opening of openings) {
  const game = new Chess();
  replay(game, opening);
  const move = await chooseMove(game, evaluate);
  if (!move) throw new Error('Engine returned no move in an open position.');
  game.move(move);
}

const match = {
  schema: 1,
  moves: ['f2f3', 'e7e5', 'g2g4', 'd8h4'],
  side: 'w',
  result: 'b',
  model: '0'.repeat(64),
};
const record = (value: object) =>
  `\`\`\`json\n${JSON.stringify(value)}\n\`\`\``;
if (
  !parseSubmission(record(match)) ||
  parseSubmission(record({ ...match, result: 'w' }))
) {
  throw new Error('Submission replay failed to verify game results.');
}
console.log(
  'Validated finite scores, legal moves, mate in one, and game submissions.',
);

const pieceValues = [
  material.p,
  material.n,
  material.b,
  material.r,
  material.q,
  material.k,
];
async function evaluateMaterial(features: Float32Array) {
  const scores = new Float32Array(64);
  for (let row = 0; row < scores.length; row++) {
    for (let i = 0; i < 768; i++) {
      const plane = Math.floor(i / 64);
      scores[row] +=
        features[row * INPUTS + i] *
        pieceValues[plane % 6] *
        (plane < 6 ? 1 : -1);
    }
    scores[row] /= 1200;
  }
  return scores;
}

if (mode === '--matches') {
  const rng = random(1991);
  for (const opponent of ['random', 'material'] as const) {
    const outcomes = { win: 0, loss: 0, draw: 0, unfinished: 0 };
    const times: number[] = [];
    const unfinishedMaterial: number[] = [];
    for (const opening of openings) {
      for (const color of ['w', 'b'] as Color[]) {
        const game = new Chess();
        replay(game, opening);
        for (let turn = 0; turn < 160 && !game.isGameOver(); turn++) {
          const start = performance.now();
          const legal =
            game.turn() === color || opponent === 'material'
              ? undefined
              : game.moves({ verbose: true });
          const move = legal
            ? legal[Math.floor(rng() * legal.length)]
            : await chooseMove(
                game,
                game.turn() === color ? evaluate : evaluateMaterial,
              );
          if (game.turn() === color) times.push(performance.now() - start);
          if (!move) throw new Error('Missing match move.');
          game.move(move);
        }
        const final = result(game);
        if (!final) {
          let balance = 0;
          for (const piece of game.board().flat()) {
            if (piece)
              balance +=
                material[piece.type] * (piece.color === color ? 1 : -1);
          }
          unfinishedMaterial.push(balance);
        }
        outcomes[
          !final
            ? 'unfinished'
            : final === 'draw'
              ? 'draw'
              : final === color
                ? 'win'
                : 'loss'
        ]++;
      }
    }
    times.sort((a, b) => a - b);
    console.log(
      `${opponent}: ${JSON.stringify(outcomes)}. Unfinished material balances (cp): ${unfinishedMaterial.join(', ')}. Engine move P95 ${times[Math.ceil(times.length * 0.95) - 1].toFixed(1)} ms.`,
    );
  }
}

process.exit(0);
