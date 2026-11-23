<script lang="ts">
import type { Color } from 'chess.js';

let { active, onstart }: { active: boolean; onstart: (side: Color) => void } =
  $props();
let open = $state(false);
let toggle: HTMLElement;

function close() {
  open = false;
  toggle.focus();
}
</script>

<details class="game-disclosure" bind:open>
  <summary bind:this={toggle}>New game</summary>
  {#if open}
    {#if active}
      <p class="note">Starting a new game replaces the current game.</p>
    {/if}
    <div class="actions">
      {#each [
        { side: 'w' as const, name: 'White' },
        { side: 'b' as const, name: 'Black' },
      ] as choice (choice.side)}
        <button
          type="button"
          onclick={() => {
            close();
            onstart(choice.side);
          }}
        >
          Play as {choice.name}
        </button>
      {/each}
      <button type="button" class="quiet-button" onclick={close}>Cancel</button>
    </div>
  {/if}
</details>
