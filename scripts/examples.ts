import { Chess } from 'chess.js';
import { INPUTS } from '../src/model';
import { encode, playCoordinate } from '../src/position';
import { teach } from './teacher';

export type Example = { features: Float32Array; value: number };

export function random(seed: number) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

function sample(game: Chess, value: number): Example {
  const features = new Float32Array(INPUTS);
  encode(game, features);
  return { features, value };
}

export function generate(games: number, seed: number) {
  const rng = random(seed);
  const samples: Example[] = [];
  for (let index = 0; index < games; index++) {
    const game = new Chess();
    for (let ply = 0; ply < 52 && !game.isGameOver(); ply++) {
      const choices = game.moves({ verbose: true });
      const { move, value } = teach(game, choices);
      if (!move) break;
      if (ply >= 6) samples.push(sample(game, value));
      game.move(
        ply < 8 || rng() < 0.35
          ? choices[Math.floor(rng() * choices.length)]
          : move,
      );
    }
  }
  return samples;
}

export function fromGames(games: readonly { moves: string[] }[]) {
  const samples: Example[] = [];
  for (const { moves } of games) {
    const game = new Chess();
    const stride = Math.max(2, Math.ceil((moves.length - 2) / 8));
    for (const [ply, move] of moves.entries()) {
      if (ply >= 2 && (ply - 2) % stride === 0)
        samples.push(sample(game, teach(game).value));
      playCoordinate(game, move);
    }
  }
  return samples;
}
