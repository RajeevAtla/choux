<script lang="ts">
import { Chess, type Color, type Move, type Square } from 'chess.js';
import { onMount } from 'svelte';
import Board from './Board.svelte';
import EngineTools from './EngineTools.svelte';
import type { EngineRequest, EngineResponse } from './engineMessages';
import MoveHistory from './MoveHistory.svelte';
import NewGame from './NewGame.svelte';
import { coordinate, playCoordinate, replay, result } from './position';
import { maxGamePlies, submissionUrl } from './submission';
import ThemePicker from './ThemePicker.svelte';

let game = new Chess();
let worker: Worker;
let id = 0;
const checkpoint = new URL(
  `${import.meta.env.BASE_URL}model.f32`,
  location.href,
).href;
const search = new URLSearchParams(location.search).get('search');
let side = $state<Color>('w');
let pieces = $state(game.board().flat());
let moves = $state<string[]>([]);
let notation = $state<string[]>([]);
let reviewPly = $state<number | null>(null);
const shownPly = $derived(reviewPly ?? moves.length);
const displayedPieces = $derived.by(() => {
  if (reviewPly === null) return pieces;
  const snapshot = new Chess();
  replay(snapshot, moves.slice(0, reviewPly));
  return snapshot.board().flat();
});
let selected = $state<Square>();
let legal = $state<Move[]>([]);
let promotion = $state<Move[]>([]);
let model = $state('');
let status = $state('Loading model…');
let outcome = $state<ReturnType<typeof result>>();
let playerTurn = $state(false);
let engineOpen = $state(false);
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
    send({
      type: 'move',
      checkpoint,
      id: ++id,
      moves: [...moves],
      search: search === 'mctx' || search === 'tactical' ? search : undefined,
    });
  }
}

const saveKey = 'choux-game-v1';

function saveGame() {
  try {
    localStorage.setItem(saveKey, JSON.stringify({ side, moves }));
  } catch {
    // Play remains available when browser storage is unavailable.
  }
}

function play(move: string) {
  const played = playCoordinate(game, move);
  moves = [...moves, move];
  saveGame();
  notation = [...notation, played.san];
  pieces = game.board().flat();
  selected = undefined;
  legal = [];
  promotion = [];
  updateTurn();
}

function select(square: Square) {
  if (!playerTurn || promotion.length || reviewPly !== null) return;
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

function review(ply: number | null) {
  reviewPly = ply;
  selected = undefined;
  legal = [];
  promotion = [];
}

function start(
  nextSide: Color,
  previous = new Chess(),
  previousMoves: string[] = [],
) {
  worker?.terminate();
  worker = new Worker(new URL('./engine.worker.ts', import.meta.url), {
    type: 'module',
  });
  side = nextSide;
  game = previous;
  pieces = game.board().flat();
  moves = previousMoves;
  notation = game.history();
  saveGame();
  reviewPly = null;
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
  let saved: { side: Color; game: Chess; moves: string[] } | undefined;
  try {
    const value = JSON.parse(localStorage.getItem(saveKey) ?? 'null');
    if (
      (value?.side === 'w' || value?.side === 'b') &&
      Array.isArray(value.moves)
    ) {
      const restored = new Chess();
      replay(restored, value.moves);
      saved = { side: value.side, game: restored, moves: value.moves };
    }
  } catch {
    // Discard an unreadable or illegal saved game.
  }
  if (saved) start(saved.side, saved.game, saved.moves);
  else start('w');
  return () => worker.terminate();
});
</script>

<main class="game">
  <header>
    <h1>Choux</h1>
    <ThemePicker />
  </header>
  <div
    class="game-status"
    data-turn={playerTurn && reviewPly === null ? 'player' : 'choux'}
  >
    <p role="status">
      {#if reviewPly !== null}
        Reviewing move {Math.ceil(reviewPly / 2)} · {notation[reviewPly - 1]}
      {:else if promotion.length}
        Choose a promotion piece.
      {:else}
        {#if notation.length}
          <span class="sr-only">Last move {notation.at(-1)}. </span>
        {/if}
        {status}
      {/if}
    </p>
    <p class="side">Playing {side === 'w' ? 'White' : 'Black'}</p>
  </div>
  <div class="game-layout">
    <div class="board-area">
      <Board
        pieces={displayedPieces}
        {side}
        {selected}
        destinations={legal.map((move) => move.to)}
        lastMove={moves[shownPly - 1]}
        interactive={playerTurn && !promotion.length && reviewPly === null}
        onselect={select}
        oncancel={() => review(null)}
      />
      <p id="board-keys" class="keyboard-hint note">
        Arrow keys to focus, Enter to select or move, Escape to clear.
      </p>
      <p class="sr-only" aria-live="polite">
        {selected
          ? `Selected ${selected}. Legal destinations: ${[...new Set(legal.map((move) => move.to))].join(', ')}.`
          : ''}
      </p>
    </div>
    <section aria-label="Game controls" class="game-controls">
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
      {#if notation.length}
        <MoveHistory {notation} {reviewPly} onreview={review} />
      {/if}
      <NewGame active={moves.length > 0 && !outcome} onstart={start} />
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
  <div class="engine-footer">
    <details class="game-disclosure" bind:open={engineOpen}>
      <summary>Engine &amp; model</summary>
      <EngineTools checkpoint={model} active={engineOpen} />
    </details>
    <span class="note">Local engine · Wasm</span>
  </div>
</main>
