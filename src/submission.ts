import type { Color } from 'chess.js';

export const submissionTitle = 'Choux training game';
export const maxGamePlies = 512;

export type SubmittedGame = {
  schema: 1;
  moves: string[];
  side: Color;
  result: Color | 'draw';
  model: string;
};

export function submissionUrl(game: SubmittedGame) {
  const url = new URL('https://github.com/RajeevAtla/choux/issues/new');
  url.search = new URLSearchParams({
    title: submissionTitle,
    body: `\`\`\`json\n${JSON.stringify(game)}\n\`\`\``,
  }).toString();
  return url.href;
}
