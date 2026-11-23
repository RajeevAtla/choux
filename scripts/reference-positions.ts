import { Chess } from 'chess.js';

const openings = [
  ['Danish gambit', 'e4 e5 d4 exd4 c3 dxc3 Bc4 cxb2 Bxb2'],
  ['Najdorf', 'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nf3 Be7'],
  ['Queens gambit', 'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7'],
  ['French defense', 'e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7'],
  ['Kings indian', 'd4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7'],
  ['Ruy Lopez', 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O'],
  ['Evans gambit', 'e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O'],
  ['Caro-Kann', 'e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7'],
  ['English opening', 'c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nf3 Nc6 O-O Be7'],
  ['Slav defense', 'd4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4'],
];

// Freeze this suite so older and newer checkpoints face the same positions.
export const referencePositions = openings
  .flatMap(([name, notation]) => {
    const moves = notation.split(' ');
    const game = new Chess();
    for (const move of moves.slice(0, -1)) game.move(move);
    const first = { name: `${name} · ${game.turn()}`, fen: game.fen() };
    game.move(moves[moves.length - 1]);
    return [first, { name: `${name} · ${game.turn()}`, fen: game.fen() }];
  })
  .concat([
    { name: 'Pawn ending · w', fen: '8/5pk1/6p1/8/4P3/6P1/5PK1/8 w - - 0 35' },
    {
      name: 'Rook ending · b',
      fen: '8/5p1k/4p1p1/7p/3R3P/6P1/5PK1/4r3 b - - 0 35',
    },
  ]);
