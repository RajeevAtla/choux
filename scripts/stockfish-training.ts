import { Chess } from 'chess.js';
import { evaluator, rankMoves } from '../src/engine';
import { coordinate, playCoordinate } from '../src/position';
import {
  positionKey,
  random,
  sample,
  sampledPly,
  type TrainingGame,
} from './examples';
import { nodes, startStockfish } from './stockfish';

const trainingNodes = 8_000;
const positionsPerSource = 96;
const mistakeLoss = 0.1;
const validationOpenings = [
  'd4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5',
  'e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 e6',
  'e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5',
  'd4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O Bf4 d6 e3 Nbd7',
  'Nf3 d5 g3 c5 Bg2 Nc6 O-O e5 d3 Nf6 e4 Be7',
  'e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6',
  'e4 c6 d3 d5 Nd2 e5 Ngf3 Bd6 Be2 Nf6 O-O O-O',
  'd4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6',
];

// Include favorable and unfavorable middlegames and endings along with openings.
const validationLaterPositions = [
  'r2qk1n1/p1pp1p2/n7/4p3/2P5/3PP2R/P2K1Pb1/R4BN1 w q - 0 13',
  '1n6/5k2/r2p4/3p3p/PP6/R7/3PK1n1/2B5 w - - 2 25',
  '1nbk1q2/r1p1p1p1/pp1pp3/8/8/2P5/PP1PPK1P/RNB2B1R w - - 0 13',
  '3k4/5pp1/3p4/2r5/p5PR/P1K5/P5P1/5B2 w - - 0 25',
  '4k3/3p4/8/p2Ppp2/1p6/3PP3/4NP2/2K5 w - - 0 25',
  'rn1q1k1r/p4p1p/1Q1pp3/5n2/8/4PN2/PP1P2P1/RNB1KB2 w Q - 1 13',
  'r2k2n1/p1pp1pp1/b1n5/3p4/2p5/P4P2/P3P1Pr/2KR1BNR w - - 0 13',
  '4kb1r/4n3/N4p2/7p/1P6/2P3KP/P5P1/R7 w k - 0 25',
  '4kbnr/1N3ppp/p5b1/2p1p3/8/2PP2KP/PP4P1/R1B2BR1 b k - 0 17',
  '8/5k2/5pn1/3N4/PP5p/7P/8/R6K b - - 0 31',
  '5k2/8/6P1/3R4/p2K4/P7/P5P1/8 b - - 0 31',
  '2rk4/p1pp1pp1/b7/3p4/2p1R1n1/P4P2/P2KP1PR/5BN1 b - - 4 17',
  '6k1/8/3p4/3P1p2/1p1P1P2/p2P4/8/4K3 b - - 5 31',
  'k1b2b2/1Q2pp2/8/p1pp4/P7/8/2PPP3/5K2 b - - 3 31',
  'q4b2/1p1pp1k1/8/p5N1/2r1n3/4P3/PPPP1PP1/1RBQK3 b - - 0 17',
  '8/1k6/8/3p1p2/Pp1B4/3bK1P1/5P2/8 b - - 3 31',
];

export const stockfishValidationFens = validationOpenings
  .flatMap((line) => {
    const game = new Chess();
    return line.split(' ').flatMap((move, ply) => {
      game.move(move);
      return ply === 6 || ply === 7 || ply === 10 || ply === 11
        ? [game.fen()]
        : [];
    });
  })
  .concat(validationLaterPositions);

function replayPositions(
  games: readonly TrainingGame[],
  seen: Set<string>,
  seed: number,
) {
  const positions: string[] = [];
  for (const { moves } of games) {
    const game = new Chess();
    for (const [ply, move] of moves.entries()) {
      if (sampledPly(ply, moves.length) && !game.isGameOver()) {
        const fen = game.fen();
        const key = positionKey(fen);
        if (!seen.has(key)) {
          positions.push(fen);
          seen.add(key);
        }
      }
      playCoordinate(game, move);
    }
  }
  const rng = random(seed);
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [positions[i], positions[j]] = [positions[j], positions[i]];
  }
  return positions;
}

export async function labelWithStockfish(
  executable: string,
  incumbent: ArrayBuffer,
  issueGames: readonly TrainingGame[],
  selfPlayGames: readonly TrainingGame[],
  excluded: ReadonlySet<string>,
) {
  const seen = new Set(excluded);
  const issuePositions = replayPositions(issueGames, seen, 162629);
  const selfPlayPositions = replayPositions(selfPlayGames, seen, 287939);
  const stockfish = await startStockfish(executable);
  try {
    const evaluate = await evaluator(incumbent);
    const labels = new Map<string, number>();
    const policyPositions: { fen: string; bestMove: string }[] = [];
    let screened = 0;
    let mistakes = 0;
    let ordinary = 0;
    for (let index = 0; index < positionsPerSource; index++) {
      for (const fen of [issuePositions[index], selfPlayPositions[index]]) {
        if (!fen) continue;
        const game = new Chess(fen);
        const predicted = coordinate((await rankMoves(game, evaluate))[0].move);
        const options = { nodes: trainingNodes, moves: [] };
        const best = await stockfish.analyze(fen, undefined, options);
        const played =
          best.move === predicted
            ? best
            : await stockfish.analyze(fen, predicted, options);
        screened++;
        const mistake = best.points - played.points >= mistakeLoss;
        if (!mistake && index % 3 !== 0) continue;
        if (mistake) mistakes++;
        else ordinary++;
        labels.set(positionKey(fen), 2 * best.points - 1);
        policyPositions.push({ fen, bestMove: best.move });
      }
    }
    const validation = [];
    for (const fen of stockfishValidationFens) {
      const { points } = await stockfish.analyze(fen, undefined, {
        nodes,
        moves: [],
      });
      validation.push(sample(new Chess(fen), 2 * points - 1, true));
    }
    return {
      labels,
      validation,
      policyPositions,
      screened,
      mistakes,
      ordinary,
    };
  } finally {
    stockfish.close();
  }
}
