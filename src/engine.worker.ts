import { Chess } from 'chess.js';
import { chooseMove, chooseMoveTactical, evaluator } from './engine';
import type { EngineRequest, EngineResponse } from './engineMessages';
import { coordinate, replay } from './position';

let loading: ReturnType<typeof load> | undefined;

async function load(url: string) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Checkpoint request failed (${response.status}).`);
  const checkpoint = await response.arrayBuffer();
  const digest = new Uint8Array(
    await crypto.subtle.digest('SHA-256', checkpoint),
  );
  const model = Array.from(digest, (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
  return { evaluate: await evaluator(checkpoint), model };
}

self.onmessage = async ({ data }: MessageEvent<EngineRequest>) => {
  function send(result: EngineResponse) {
    self.postMessage(result);
  }
  try {
    loading ??= load(data.checkpoint);
    const { evaluate, model } = await loading;
    if (data.type === 'load') {
      send({ type: 'ready', id: data.id, model });
    } else {
      const game = new Chess();
      replay(game, data.moves);
      const search = data.tactical ? chooseMoveTactical : chooseMove;
      const move = await search(game, evaluate);
      if (!move) throw new Error('No legal move remains.');
      send({ type: 'move', id: data.id, move: coordinate(move) });
    }
  } catch (error) {
    send({
      type: 'error',
      id: data.id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
