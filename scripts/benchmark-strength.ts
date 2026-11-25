import { Chess } from 'chess.js';
import {
  chooseMove,
  chooseMoveBaseline,
  chooseMoveTactical,
  evaluator,
  moveBudgetMs,
} from '../src/engine';
import { coordinate } from '../src/position';
import { referencePositions } from './reference-positions';
import { nodes, startStockfish } from './stockfish';

type ScoredMove = {
  move: string;
  loss: number;
  match: boolean;
  elapsedMs: number;
};
type Comparison = {
  name: string;
  fen: string;
  stockfish: { move: string; points: number };
  candidate: ScoredMove;
  incumbent: ScoredMove;
};

const [candidatePath, incumbentPath, output, mode] = Bun.argv.slice(2);
if (
  !candidatePath ||
  !incumbentPath ||
  !output ||
  (mode && mode !== '--search-ab')
) {
  throw new Error(
    'Usage: bun run benchmark:strength <candidate> <incumbent> <report> [--search-ab]',
  );
}

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
if (mode && candidate.hash !== incumbent.hash)
  throw new Error('Search comparison requires identical checkpoint weights.');

const incumbentSearch = mode ? chooseMoveBaseline : chooseMove;
const candidateSearch = mode ? chooseMoveTactical : chooseMove;
async function timedSearch(
  search: typeof chooseMove,
  game: Chess,
  evaluate: Awaited<ReturnType<typeof evaluator>>,
) {
  const start = performance.now();
  const move = await search(game, evaluate, moveBudgetMs);
  if (!move) throw new Error(`No move from ${game.fen()}.`);
  return { move: coordinate(move), elapsedMs: performance.now() - start };
}
const stockfish = await startStockfish(Bun.env.STOCKFISH ?? 'stockfish');
try {
  const evaluateCandidate = await evaluator(candidate.bytes);
  if (mode) evaluateCandidate.movePrior = undefined;
  const evaluateIncumbent =
    candidate.hash === incumbent.hash
      ? evaluateCandidate
      : await evaluator(incumbent.bytes);
  const positions: Comparison[] = [];
  for (const { name, fen } of referencePositions) {
    const game = new Chess(fen);
    const candidateMove = await timedSearch(
      candidateSearch,
      game,
      evaluateCandidate,
    );
    const incumbentMove =
      !mode && evaluateIncumbent === evaluateCandidate
        ? candidateMove
        : await timedSearch(incumbentSearch, game, evaluateIncumbent);

    const best = await stockfish.analyze(fen);
    const pointsByMove = new Map([[best.move, best.points]]);
    async function scoreMove({ move, elapsedMs }: typeof candidateMove) {
      let points = pointsByMove.get(move);
      if (points === undefined) {
        points = (await stockfish.analyze(fen, move)).points;
        pointsByMove.set(move, points);
      }
      return {
        move,
        loss: Math.max(0, best.points - points),
        match: move === best.move,
        elapsedMs,
      };
    }
    positions.push({
      name,
      fen,
      stockfish: best,
      candidate: await scoreMove(candidateMove),
      incumbent: await scoreMove(incumbentMove),
    });
  }

  function summary(key: 'candidate' | 'incumbent', hash: string) {
    const total = positions.reduce(
      (sum, position) => sum + position[key].loss,
      0,
    );
    return {
      hash,
      meanLoss: Number(((100 * total) / positions.length).toFixed(2)),
      bestMoveMatches: positions.filter((position) => position[key].match)
        .length,
    };
  }
  const report = {
    suite: 'choux-reference-v2',
    moveBudgetMs,
    mode: mode ? 'search-ab' : 'checkpoints',
    engine: stockfish.name,
    nodes,
    positionCount: positions.length,
    candidate: summary('candidate', candidate.hash),
    incumbent: summary('incumbent', incumbent.hash),
    positions,
  };
  await Bun.write(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(
    `${report.engine}, ${report.positionCount} positions. Mean expected-score loss: candidate ${report.candidate.meanLoss} pp, incumbent ${report.incumbent.meanLoss} pp.`,
  );
} finally {
  stockfish.close();
}

process.exit(0);
