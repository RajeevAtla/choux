<script lang="ts">
import { Chess, type Color, type Move, type Square } from 'chess.js';
import { onMount } from 'svelte';
import Board from './Board.svelte';
import type { EngineRequest, EngineResponse } from './engineMessages';
import { coordinate, playCoordinate, result } from './position';
import { maxGamePlies, submissionUrl } from './submission';

let game = new Chess();
let worker: Worker;
let id = 0;
const checkpoint = new URL(
  `${import.meta.env.BASE_URL}model.f32`,
  location.href,
).href;
let side = $state<Color>('w');
let pieces = $state(game.board().flat());
let moves = $state<string[]>([]);
let notation = $state<string[]>([]);
let selected = $state<Square>();
let legal = $state<Move[]>([]);
let promotion = $state<Move[]>([]);
let model = $state('');
let status = $state('Loading model…');
let outcome = $state<ReturnType<typeof result>>();
let playerTurn = $state(false);
const promotionNames: Record<string, string> = {
  q: 'Queen',
  r: 'Rook',
  b: 'Bishop',
  n: 'Knight',
};

function send(message: EngineRequest) {
  worker.postMessage(message);
}

function updateTurn() {
  outcome = result(game);
  playerTurn = !!model && !outcome && game.turn() === side;
  if (outcome) {
    status =
      outcome === 'draw'
        ? 'Draw.'
        : `${outcome === 'w' ? 'White' : 'Black'} wins by checkmate.`;
  } else if (playerTurn) {
    status = game.isCheck() ? 'Your move. You are in check.' : 'Your move.';
  } else if (model) {
    status = 'Choux is thinking…';
    send({ type: 'move', checkpoint, id: ++id, moves: [...moves] });
  }
}

function play(move: string) {
  const played = playCoordinate(game, move);
  moves = [...moves, move];
  notation = [...notation, played.san];
  pieces = game.board().flat();
  selected = undefined;
  legal = [];
  promotion = [];
  updateTurn();
}

function select(square: Square) {
  if (!playerTurn || promotion.length) return;
  const choices = legal.filter((move) => move.to === square);
  if (choices.length > 1) {
    promotion = choices;
  } else if (choices.length === 1) {
    play(coordinate(choices[0]));
  } else {
    legal = square === selected ? [] : game.moves({ square, verbose: true });
    selected = legal.length ? square : undefined;
  }
}

function start(nextSide: Color) {
  worker?.terminate();
  worker = new Worker(new URL('./engine.worker.ts', import.meta.url), {
    type: 'module',
  });
  side = nextSide;
  game = new Chess();
  pieces = game.board().flat();
  moves = [];
  notation = [];
  legal = [];
  promotion = [];
  selected = undefined;
  model = '';
  playerTurn = false;
  outcome = undefined;
  status = 'Loading model…';
  const current = worker;
  worker.onmessage = ({ data }: MessageEvent<EngineResponse>) => {
    if (current !== worker || data.id !== id) return;
    if (data.type === 'error') status = data.message;
    else if (data.type === 'ready') {
      model = data.model;
      updateTurn();
    } else {
      play(data.move);
    }
  };
  worker.onerror = () => {
    if (current === worker)
      status = 'The engine could not start. Try a new game.';
  };
  send({ type: 'load', checkpoint, id: ++id });
}

onMount(() => {
  start('w');
  return () => worker.terminate();
});
</script>

<main class="game">
  <header>
    <h1>Choux</h1>
    <a class="benchmark-link" href="?benchmark">Inference benchmark</a>
  </header>
  <div class="game-status" data-turn={playerTurn ? 'player' : 'choux'}>
    <p role="status">{status}</p>
    <p class="side">Playing {side === 'w' ? 'White' : 'Black'}</p>
  </div>
  <div class="game-layout">
    <Board
      {pieces}
      {side}
      {selected}
      destinations={legal.map((move) => move.to)}
      lastMove={moves.at(-1)}
      interactive={playerTurn && !promotion.length}
      onselect={select}
    />
    <section aria-label="Game controls" class="game-controls">
      <h2>New game</h2>
      <div class="actions">
        <button type="button" onclick={() => start('w')}>Play White</button>
        <button type="button" onclick={() => start('b')}>Play Black</button>
      </div>
      {#if promotion.length}
        <fieldset class="promotion">
          <legend>Promote pawn</legend>
          {#each promotion as move (move.promotion)}
            <button type="button" onclick={() => play(coordinate(move))}>
              {promotionNames[move.promotion ?? 'q']}
            </button>
          {/each}
        </fieldset>
      {/if}
      <h2 class="moves-heading">Moves</h2>
      {#if notation.length}
        <p class="moves">
          {notation
            .map((move, index) =>
              index % 2 ? move : `${1 + index / 2}. ${move}`,
            )
            .join(' ')}
        </p>
      {:else}
        <p class="note moves-empty">Moves appear here as you play.</p>
      {/if}
      {#if outcome && moves.length <= maxGamePlies}
        <a
          href={submissionUrl({
            schema: 1,
            moves,
            side,
            result: outcome,
            model,
          })}
          target="_blank"
          rel="noopener noreferrer"
          >Share game for training</a
        >
        <p class="note">
          Opens a public GitHub issue with your moves, result, side, and model
          version.
        </p>
      {/if}
    </section>
  </div>
</main>
