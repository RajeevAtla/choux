import { Chess } from 'chess.js';
import { replay, result } from '../src/position';
import { maxGamePlies, type SubmittedGame } from '../src/submission';

export const datasetPath = 'data/games.jsonl';

export async function readGames<T extends { moves: string[] } = SubmittedGame>(
  path = datasetPath,
): Promise<T[]> {
  const file = Bun.file(path);
  if (!(await file.exists())) return [];
  return (await file.text())
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function parseSubmission(body: string): SubmittedGame | undefined {
  if (body.length > 6000) return undefined;
  const match = /^\x60{3}json\r?\n([\s\S]+)\r?\n\x60{3}$/.exec(body.trim());
  if (!match) return undefined;
  try {
    const {
      schema,
      moves,
      side,
      result: claimedResult,
      model,
    } = JSON.parse(match[1]);
    if (
      schema !== 1 ||
      (side !== 'w' && side !== 'b') ||
      typeof model !== 'string' ||
      !/^[a-f0-9]{64}$/.test(model) ||
      !Array.isArray(moves) ||
      !moves.length ||
      moves.length > maxGamePlies ||
      !moves.every((move) => typeof move === 'string')
    )
      return undefined;

    const game = new Chess();
    replay(game, moves);
    const actualResult = result(game);
    if (!actualResult || actualResult !== claimedResult) return undefined;
    return { schema, moves, side, result: actualResult, model };
  } catch {
    return undefined;
  }
}
