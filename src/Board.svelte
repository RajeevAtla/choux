<script lang="ts">
import type { Color, Piece, Square } from 'chess.js';
import './board.css';

let {
  pieces,
  side,
  selected,
  destinations,
  lastMove,
  interactive,
  onselect,
}: {
  pieces: (Piece | null)[];
  side: Color;
  selected?: Square;
  destinations: Square[];
  lastMove?: string;
  interactive: boolean;
  onselect: (square: Square) => void;
} = $props();

const symbols = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
const names = {
  k: 'king',
  q: 'queen',
  r: 'rook',
  b: 'bishop',
  n: 'knight',
  p: 'pawn',
};
const topDown = Array.from({ length: 64 }, (_, index) => ({
  index,
  square: `${'abcdefgh'[index % 8]}${8 - Math.floor(index / 8)}` as Square,
}));
const bottomUp = [...topDown].reverse();
const squares = $derived(side === 'w' ? topDown : bottomUp);
let board: HTMLFieldSetElement;
let focused = $state<Square>();
const entry = $derived(focused ?? (side === 'w' ? 'e2' : 'e7'));

function navigate(event: KeyboardEvent, index: number) {
  const row = Math.floor(index / 8);
  const file = index % 8;
  const steps: Record<string, number | undefined> = {
    ArrowLeft: file > 0 ? -1 : 0,
    ArrowRight: file < 7 ? 1 : 0,
    ArrowUp: row > 0 ? -8 : 0,
    ArrowDown: row < 7 ? 8 : 0,
  };
  const step = steps[event.key];
  if (step === undefined) return;
  event.preventDefault();
  board.querySelectorAll('button')[index + step]?.focus();
}
</script>

<fieldset
  class="board"
  aria-label={`Chessboard, ${side === 'w' ? 'White' : 'Black'} at the bottom`}
  bind:this={board}
>
  {#each squares as { index, square }, display (square)}
    {@const piece = pieces[index]}
    <button
      type="button"
      class="square"
      class:dark={(index + Math.floor(index / 8)) % 2 !== 0}
      class:target={destinations.includes(square)}
      class:capture={!!piece && destinations.includes(square)}
      class:last={lastMove?.slice(0, 2) === square ||
        lastMove?.slice(2, 4) === square}
      aria-label={piece
        ? `${square}, ${piece.color === 'w' ? 'White' : 'Black'} ${names[piece.type]}`
        : square}
      aria-pressed={selected === square}
      aria-disabled={!interactive}
      tabindex={entry === square ? 0 : -1}
      onfocus={() => (focused = square)}
      onkeydown={(event) => navigate(event, display)}
      onclick={() => onselect(square)}
    >
      {#if display % 8 === 0}
        <span class="rank" aria-hidden="true">{square[1]}</span>
      {/if}
      {#if display >= 56}
        <span class="file" aria-hidden="true">{square[0]}</span>
      {/if}
      <span
        class="piece"
        class:light-piece={piece?.color === 'w'}
        aria-hidden="true"
        >{piece ? symbols[piece.type] : ''}</span
      >
    </button>
  {/each}
</fieldset>
