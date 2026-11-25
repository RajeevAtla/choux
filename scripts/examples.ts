import { Chess } from 'chess.js';
import { INPUTS } from '../src/model';
import { encode, playCoordinate } from '../src/position';
import { teach } from './teacher';

export type Example = {
  features: Float32Array;
  value: number;
  stockfish?: boolean;
};
export type TrainingGame = {
  moves: string[];
  stockfishValues?: Record<number, number>;
};

export function random(seed: number) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

export function sample(game: Chess, value: number, stockfish = false): Example {
  const features = new Float32Array(INPUTS);
  encode(game, features);
  return { features, value, stockfish };
}

export function positionKey(fen: string) {
  return fen.split(' ', 4).join(' ');
}

export function generate(
  games: number,
  seed: number,
  excluded?: ReadonlySet<string>,
) {
  const rng = random(seed);
  const samples: Example[] = [];
  for (let index = 0; index < games; index++) {
    const game = new Chess();
    for (let ply = 0; ply < 52 && !game.isGameOver(); ply++) {
      const choices = game.moves({ verbose: true });
      const { move, value } = teach(game, choices);
      if (!move) break;
      if (ply >= 6 && !excluded?.has(positionKey(game.fen())))
        samples.push(sample(game, value));
      game.move(
        ply < 8 || rng() < 0.35
          ? choices[Math.floor(rng() * choices.length)]
          : move,
      );
    }
  }
  return samples;
}

export function sampledPly(ply: number, length: number) {
  const stride = Math.max(2, Math.ceil((length - 2) / 8));
  return ply >= 2 && (ply - 2) % stride === 0;
}

export function fromGames(
  games: readonly TrainingGame[],
  excluded?: ReadonlySet<string>,
  stockfishLabels?: ReadonlyMap<string, number>,
) {
  const samples: Example[] = [];
  for (const { moves, stockfishValues } of games) {
    const game = new Chess();
    for (const [ply, move] of moves.entries()) {
      if (
        sampledPly(ply, moves.length) &&
        !excluded?.has(positionKey(game.fen()))
      ) {
        const value =
          stockfishLabels?.get(positionKey(game.fen())) ??
          stockfishValues?.[ply];
        samples.push(
          sample(game, value ?? teach(game).value, value !== undefined),
        );
      }
      playCoordinate(game, move);
    }
  }
  return samples;
}
