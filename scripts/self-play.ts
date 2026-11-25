import { Chess, type Color, type Move } from 'chess.js';
import { chooseMove, evaluator } from '../src/engine';
import { coordinate, playCoordinate, result } from '../src/position';
import { maxGamePlies } from '../src/submission';
import { random, sampledPly } from './examples';
import { startStockfish } from './stockfish';

const [
  output = 'self-play/games.jsonl',
  checkpointPath = 'public/model.f32',
  gameCount = '16',
  plyLimit = '96',
  seed = new Date().toISOString().slice(0, 10),
  stockfishCount = '0',
] = Bun.argv.slice(2);

const games = Number(gameCount);
const plies = Number(plyLimit);
const stockfishGames = Number(stockfishCount);
const openingPlies = 6;
const stockfishNodes = 4_000;
const trainingOpenings = [
  'e4 e5 Nf3 Nc6 Bc4 Nf6',
  'e4 e5 Nf3 Nc6 d4 exd4',
  'd4 d5 Bf4 Nf6 e3 c5',
  'e4 c5 c3 Nf6 e5 Nd5',
];
if (!Number.isInteger(games) || games < 1 || games > 64) {
  throw new Error('Game count must be an integer from 1 to 64.');
}
if (!Number.isInteger(plies) || plies <= openingPlies || plies > maxGamePlies) {
  throw new Error(
    `Ply limit must be an integer from ${openingPlies + 1} to ${maxGamePlies}.`,
  );
}
if (
  !Number.isInteger(stockfishGames) ||
  stockfishGames < 0 ||
  stockfishGames > games
) {
  throw new Error(
    'Stockfish game count must be an integer from 0 to game count.',
  );
}

const checkpoint = await Bun.file(checkpointPath).arrayBuffer();
const model = new Bun.CryptoHasher('sha256').update(checkpoint).digest('hex');
const seedHash = new Bun.CryptoHasher('sha256').update(seed).digest();
const rng = random(seedHash.readUInt32LE(0));
const openingOffset = seedHash.readUInt32LE(4) % trainingOpenings.length;
const evaluate = await evaluator(checkpoint);
const records = [];
const stockfish = stockfishGames
  ? await startStockfish(Bun.env.STOCKFISH ?? 'stockfish')
  : undefined;

try {
  for (let index = 0; index < games; index++) {
    const game = new Chess();
    const moves: string[] = [];
    const opponent = index < stockfishGames ? stockfish : undefined;
    const stockfishSide: Color = index % 2 === 0 ? 'b' : 'w';
    const opening =
      trainingOpenings[
        (openingOffset + Math.floor(index / 2)) % trainingOpenings.length
      ].split(' ');
    const playedScores = new Map<number, number>();
    while (moves.length < plies && !game.isGameOver()) {
      let played: Move;
      if (moves.length < openingPlies) {
        if (opponent) {
          played = game.move(opening[moves.length]);
        } else {
          const legal = game.moves({ verbose: true });
          played = game.move(legal[Math.floor(rng() * legal.length)]);
        }
      } else if (opponent && game.turn() === stockfishSide) {
        const { move: best, points } = await opponent.analyze(
          game.fen(),
          undefined,
          {
            nodes: stockfishNodes,
            moves,
          },
        );
        playedScores.set(moves.length, points);
        played = playCoordinate(game, best);
      } else {
        const move = await chooseMove(game, evaluate);
        if (!move)
          throw new Error('Self-play found no move in a live position.');
        played = game.move(move);
      }
      moves.push(coordinate(played));
    }
    const stockfishValues: Record<number, number> = {};
    if (opponent) {
      const replay = new Chess();
      for (const [ply, move] of moves.entries()) {
        if (ply >= openingPlies && sampledPly(ply, moves.length)) {
          const points =
            playedScores.get(ply) ??
            (
              await opponent.analyze(replay.fen(), undefined, {
                nodes: stockfishNodes,
                moves: moves.slice(0, ply),
              })
            ).points;
          stockfishValues[ply] = 2 * points - 1;
        }
        playCoordinate(replay, move);
      }
    }
    records.push({
      schema: 1,
      source: 'self-play',
      model,
      seed,
      moves,
      result: result(game) ?? null,
      ...(opponent && {
        opponent: opponent.name,
        stockfishSide,
        stockfishValues,
      }),
    });
  }
} finally {
  stockfish?.close();
}

await Bun.write(
  output,
  `${records.map((game) => JSON.stringify(game)).join('\n')}\n`,
);
const total = records.reduce((sum, game) => sum + game.moves.length, 0);
const finished = records.filter((game) => game.result !== null).length;
console.log(
  `Saved ${records.length} games (${stockfishGames} against Stockfish, ${total} plies, ${finished} finished) from ${model.slice(0, 12)} to ${output}.`,
);

// The Wasm thread pool can keep Bun alive after the games are written.
process.exit(0);
