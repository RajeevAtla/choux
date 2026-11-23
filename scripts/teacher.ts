import type { Chess, Move } from 'chess.js';

export const material = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };

export function evaluate(game: Chess) {
  let value = 0;
  for (const [rank, row] of game.board().entries()) {
    for (const [file, piece] of row.entries()) {
      if (!piece) continue;
      const advance = piece.color === 'w' ? 6 - rank : rank - 1;
      const center = 7 - Math.abs(3.5 - rank) - Math.abs(3.5 - file);
      const placement =
        piece.type === 'p'
          ? advance * 6
          : piece.type === 'k'
            ? -center * 5
            : center * 8;
      value +=
        (material[piece.type] + placement) *
        (piece.color === game.turn() ? 1 : -1);
    }
  }
  return value + (game.isCheck() ? -35 : 12);
}

export function teach(game: Chess, legal = game.moves({ verbose: true })) {
  let best: Move | undefined;
  let value = -Infinity;
  for (const move of legal) {
    game.move(move);
    const score = game.isCheckmate()
      ? 1200
      : game.isDraw()
        ? 0
        : -evaluate(game);
    game.undo();
    if (score > value) {
      value = score;
      best = move;
    }
  }
  return { move: best, value: Math.max(-1, Math.min(1, value / 1200)) };
}
