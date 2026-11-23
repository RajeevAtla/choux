import type { Chess, Color, Move } from 'chess.js';
import { INPUTS } from './modelShape';

const pieces = 'pnbrqk';

export function encode(game: Chess, output: Float32Array, offset = 0) {
  const [board, turn, castling, enPassant, halfMoves] = game.fen().split(' ');
  let square = 0;
  for (const symbol of board) {
    if (symbol === '/') continue;
    if (symbol >= '1' && symbol <= '8') {
      square += Number(symbol);
    } else {
      const isWhite = symbol === symbol.toUpperCase();
      const ownPiece = isWhite === (turn === 'w');
      const plane = pieces.indexOf(symbol.toLowerCase()) + (ownPiece ? 0 : 6);
      output[offset + plane * 64 + (turn === 'b' ? square ^ 56 : square)] = 1;
      square++;
    }
  }
  for (const [index, flag] of [...(turn === 'b' ? 'kqKQ' : 'KQkq')].entries()) {
    if (castling.includes(flag)) output[offset + 768 + index] = 1;
  }
  if (enPassant !== '-')
    output[offset + 772 + enPassant.charCodeAt(0) - 97] = 1;
  output[offset + 780] = Math.min(Number(halfMoves), 100) / 100;
  output[offset + INPUTS - 1] = turn === 'b' ? 1 : 0;
}

export function coordinate({ from, to, promotion }: Move) {
  return `${from}${to}${promotion ?? ''}`;
}

export function replay(game: Chess, moves: readonly string[]) {
  for (const move of moves) {
    if (game.isGameOver()) {
      throw new Error('Invalid game record.');
    }
    playCoordinate(game, move);
  }
}

export function playCoordinate(game: Chess, move: string) {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move))
    throw new Error('Invalid move.');
  const played = game.move({
    from: move.slice(0, 2),
    to: move.slice(2, 4),
    promotion: move[4],
  });
  if (coordinate(played) !== move) {
    game.undo();
    throw new Error('Invalid promotion.');
  }
  return played;
}

export function result(game: Chess): Color | 'draw' | undefined {
  if (game.isCheckmate()) return game.turn() === 'w' ? 'b' : 'w';
  return game.isDraw() ? 'draw' : undefined;
}
