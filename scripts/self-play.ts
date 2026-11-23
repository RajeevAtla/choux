import { Chess, type Move } from 'chess.js';
import { chooseMove, evaluator } from '../src/engine';
import { coordinate, result } from '../src/position';
import { maxGamePlies } from '../src/submission';
import { random } from './examples';

const [
  output = 'self-play/games.jsonl',
  checkpointPath = 'public/model.f32',
  gameCount = '16',
  plyLimit = '96',
  seed = new Date().toISOString().slice(0, 10),
] = Bun.argv.slice(2);

const games = Number(gameCount);
const plies = Number(plyLimit);
const openingPlies = 6;
if (!Number.isInteger(games) || games < 1 || games > 64) {
  throw new Error('Game count must be an integer from 1 to 64.');
}
if (!Number.isInteger(plies) || plies <= openingPlies || plies > maxGamePlies) {
  throw new Error(
    `Ply limit must be an integer from ${openingPlies + 1} to ${maxGamePlies}.`,
  );
}

const checkpoint = await Bun.file(checkpointPath).arrayBuffer();
const model = new Bun.CryptoHasher('sha256').update(checkpoint).digest('hex');
const seedHash = new Bun.CryptoHasher('sha256').update(seed).digest();
const rng = random(seedHash.readUInt32LE(0));
const evaluate = await evaluator(checkpoint);
const records = [];

for (let index = 0; index < games; index++) {
  const game = new Chess();
  const moves: string[] = [];
  while (moves.length < plies && !game.isGameOver()) {
    let move: Move | undefined;
    if (moves.length < openingPlies) {
      const legal = game.moves({ verbose: true });
      move = legal[Math.floor(rng() * legal.length)];
    } else {
      move = await chooseMove(game, evaluate, Infinity);
    }
    if (!move) throw new Error('Self-play found no move in a live position.');
    game.move(move);
    moves.push(coordinate(move));
  }
  records.push({
    schema: 1,
    source: 'self-play',
    model,
    seed,
    moves,
    result: result(game) ?? null,
  });
}

await Bun.write(
  output,
  `${records.map((game) => JSON.stringify(game)).join('\n')}\n`,
);
const total = records.reduce((sum, game) => sum + game.moves.length, 0);
const finished = records.filter((game) => game.result !== null).length;
console.log(
  `Saved ${records.length} games (${total} plies, ${finished} finished) from ${model.slice(0, 12)} to ${output}.`,
);

// The Wasm thread pool can keep Bun alive after the games are written.
process.exit(0);
