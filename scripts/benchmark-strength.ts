import { Chess } from 'chess.js';
import { chooseMove, evaluator } from '../src/engine';
import { coordinate } from '../src/position';
import { referencePositions } from './reference-positions';
import { nodes, startStockfish } from './stockfish';

type ScoredMove = { move: string; loss: number; match: boolean };
type Comparison = {
  name: string;
  fen: string;
  stockfish: { move: string; points: number };
  candidate: ScoredMove;
  incumbent: ScoredMove;
};

const [candidatePath, incumbentPath, output] = Bun.argv.slice(2);
if (!candidatePath || !incumbentPath || !output) {
  throw new Error(
    'Usage: bun run benchmark:strength <candidate> <incumbent> <report>',
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
const stockfish = await startStockfish(Bun.env.STOCKFISH ?? 'stockfish');
try {
  const evaluateCandidate = await evaluator(candidate.bytes);
  const evaluateIncumbent =
    candidate.hash === incumbent.hash
      ? evaluateCandidate
      : await evaluator(incumbent.bytes);
  const positions: Comparison[] = [];
  for (const { name, fen } of referencePositions) {
    const game = new Chess(fen);
    const candidateMove = await chooseMove(game, evaluateCandidate, Infinity);
    const incumbentMove =
      evaluateIncumbent === evaluateCandidate
        ? candidateMove
        : await chooseMove(game, evaluateIncumbent, Infinity);
    if (!candidateMove || !incumbentMove)
      throw new Error(`No move from ${name}.`);

    const best = await stockfish.analyze(fen);
    const pointsByMove = new Map([[best.move, best.points]]);
    async function scoreMove(move: string) {
      let points = pointsByMove.get(move);
      if (points === undefined) {
        points = (await stockfish.analyze(fen, move)).points;
        pointsByMove.set(move, points);
      }
      return {
        move,
        loss: Math.max(0, best.points - points),
        match: move === best.move,
      };
    }
    positions.push({
      name,
      fen,
      stockfish: best,
      candidate: await scoreMove(coordinate(candidateMove)),
      incumbent: await scoreMove(coordinate(incumbentMove)),
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
    suite: 'choux-reference-v1',
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
