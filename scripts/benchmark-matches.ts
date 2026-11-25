import { Chess, type Color } from 'chess.js';
import {
  chooseMove,
  chooseMoveBaseline,
  chooseMoveTactical,
  evaluator,
  moveBudgetMs,
} from '../src/engine';
import { coordinate, playCoordinate, result } from '../src/position';
import { referenceOpenings } from './reference-positions';
import { startStockfish } from './stockfish';

const [candidatePath, incumbentPath, output, pairsArg = '2', pliesArg = '200'] =
  Bun.argv.slice(2).filter((argument) => argument !== '--search-ab');
const searchAB = Bun.argv.slice(2).includes('--search-ab');
const pairs = Number(pairsArg);
const plyLimit = Number(pliesArg);
const stockfishNodes = 4_000;
if (
  !candidatePath ||
  !incumbentPath ||
  !output ||
  !Number.isInteger(pairs) ||
  pairs < 1 ||
  pairs > referenceOpenings.length ||
  !Number.isInteger(plyLimit) ||
  plyLimit < 1 ||
  plyLimit > 512
) {
  throw new Error(
    'Usage: bun run benchmark:matches <candidate> <incumbent> <report> [opening pairs] [plies after opening] [--search-ab]',
  );
}

type Player = 'candidate' | 'incumbent' | 'stockfish';
type Search = typeof chooseMove;
type Evaluation = Awaited<ReturnType<typeof evaluator>>;

async function checkpoint(path: string) {
  const bytes = await Bun.file(path).arrayBuffer();
  return {
    bytes,
    hash: new Bun.CryptoHasher('sha256').update(bytes).digest('hex'),
  };
}

const [candidate, incumbent] = await Promise.all([
  checkpoint(candidatePath),
  checkpoint(incumbentPath),
]);
if (searchAB && candidate.hash !== incumbent.hash)
  throw new Error('Search comparison requires identical checkpoint weights.');
const evaluateCandidate = await evaluator(candidate.bytes);
if (searchAB) evaluateCandidate.movePrior = undefined;
const evaluateIncumbent =
  candidate.hash === incumbent.hash
    ? evaluateCandidate
    : await evaluator(incumbent.bytes);
const searches = {
  candidate: searchAB ? chooseMoveTactical : chooseMove,
  incumbent: searchAB ? chooseMoveBaseline : chooseMove,
};
const stockfish = await startStockfish(Bun.env.STOCKFISH ?? 'stockfish');

async function chouxMove(game: Chess, search: Search, evaluate: Evaluation) {
  const started = performance.now();
  const move = await search(game, evaluate, moveBudgetMs);
  if (!move) throw new Error(`No move from ${game.fen()}.`);
  return { move: coordinate(move), elapsedMs: performance.now() - started };
}

async function play(
  openingName: string,
  opening: string,
  white: Player,
  black: Player,
) {
  const game = new Chess();
  const moves = opening.split(' ').map((move) => coordinate(game.move(move)));
  const played: Record<Exclude<Player, 'stockfish'>, number[]> = {
    candidate: [],
    incumbent: [],
  };
  const maxPlies = moves.length + plyLimit;
  while (moves.length < maxPlies && !game.isGameOver()) {
    const side = game.turn() === 'w' ? white : black;
    if (side === 'stockfish') {
      const { move } = await stockfish.analyze(game.fen(), undefined, {
        nodes: stockfishNodes,
        moves,
      });
      moves.push(coordinate(playCoordinate(game, move)));
    } else {
      const { move, elapsedMs } = await chouxMove(
        game,
        searches[side],
        side === 'candidate' ? evaluateCandidate : evaluateIncumbent,
      );
      played[side].push(elapsedMs);
      moves.push(coordinate(playCoordinate(game, move)));
    }
  }
  const outcome = result(game);
  const winner = { w: white, b: black } satisfies Record<Color, Player>;
  return {
    opening: openingName,
    white,
    black,
    moves,
    outcome:
      outcome && outcome !== 'draw' ? winner[outcome] : (outcome ?? null),
    elapsedMs: played,
  };
}

type Match = Awaited<ReturnType<typeof play>>;
function latency(samples: number[]) {
  samples.sort((a, b) => a - b);
  return {
    moves: samples.length,
    meanMs: samples.length
      ? samples.reduce((sum, sample) => sum + sample, 0) / samples.length
      : null,
    p95Ms: samples[Math.ceil(samples.length * 0.95) - 1] ?? null,
    overBudget: samples.filter((sample) => sample > moveBudgetMs).length,
  };
}

const series = [];
try {
  const pairings = [
    ['candidate', 'stockfish'],
    ['incumbent', 'stockfish'],
    ['candidate', 'incumbent'],
  ] as const;
  for (const [first, second] of pairings) {
    const games: Match[] = [];
    for (const [name, opening] of referenceOpenings.slice(0, pairs)) {
      games.push(await play(name, opening, first, second));
      games.push(await play(name, opening, second, first));
    }
    const summary = {
      wins: games.filter(({ outcome }) => outcome === first).length,
      draws: games.filter(({ outcome }) => outcome === 'draw').length,
      losses: games.filter(({ outcome }) => outcome === second).length,
      unfinished: games.filter(({ outcome }) => outcome === null).length,
    };
    series.push({ first, second, summary, games });
    console.log(
      `${first} vs ${second}: ${summary.wins}W ${summary.draws}D ${summary.losses}L, ${summary.unfinished} unfinished.`,
    );
  }
  const games = series.flatMap((pairing) => pairing.games);
  await Bun.write(
    output,
    `${JSON.stringify(
      {
        suite: 'choux-matches-v1',
        mode: searchAB ? 'search-ab' : 'checkpoints',
        candidate: candidate.hash,
        incumbent: incumbent.hash,
        stockfish: stockfish.name,
        stockfishNodes,
        moveBudgetMs,
        plyLimit,
        latency: {
          candidate: latency(
            games.flatMap(({ elapsedMs }) => elapsedMs.candidate),
          ),
          incumbent: latency(
            games.flatMap(({ elapsedMs }) => elapsedMs.incumbent),
          ),
        },
        series,
      },
      null,
      2,
    )}\n`,
  );
} finally {
  stockfish.close();
}

process.exit(0);
