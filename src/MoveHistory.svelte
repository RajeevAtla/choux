<script lang="ts">
let {
  notation,
  reviewPly,
  onreview,
}: {
  notation: string[];
  reviewPly: number | null;
  onreview: (ply: number | null) => void;
} = $props();

let sheet: HTMLDivElement;
const current = $derived(reviewPly ?? notation.length);
const turns = $derived(
  Array.from(
    { length: Math.ceil(notation.length / 2) },
    (_, index) => index * 2 + 1,
  ),
);

$effect(() => {
  if (reviewPly === null && current && sheet)
    sheet.scrollTop = sheet.scrollHeight;
});

function navigate(event: KeyboardEvent, ply: number) {
  const steps: Record<string, number | undefined> = {
    ArrowLeft: -1,
    ArrowRight: 1,
    ArrowUp: -2,
    ArrowDown: 2,
  };
  const step = steps[event.key];
  if (step === undefined) return;
  event.preventDefault();
  const next = Math.min(notation.length, Math.max(1, ply + step));
  onreview(next);
  sheet.querySelector<HTMLButtonElement>(`[data-ply="${next}"]`)?.focus();
}
</script>

<div class="history-heading">
  <h2>Moves</h2>
  {#if reviewPly !== null}
    <button
      type="button"
      class="quiet-button"
      onclick={() => {
        onreview(null);
        sheet
          .querySelector<HTMLButtonElement>(`[data-ply="${notation.length}"]`)
          ?.focus();
      }}
    >
      Back to live game
    </button>
  {/if}
</div>
<div class="moves" bind:this={sheet}>
  <table aria-label="Move history">
    <thead>
      <tr>
        <th scope="col">No.</th>
        <th scope="col">White</th>
        <th scope="col">Black</th>
      </tr>
    </thead>
    <tbody>
      {#each turns as ply (ply)}
        <tr>
          <th scope="row">{(ply + 1) / 2}</th>
          {#each [ply, ply + 1] as movePly (movePly)}
            <td>
              {#if notation[movePly - 1]}
                <button
                  type="button"
                  class="history-move"
                  class:latest={current === movePly}
                  data-ply={movePly}
                  tabindex={current === movePly ? 0 : -1}
                  aria-current={current === movePly ? 'step' : undefined}
                  aria-label={`Move ${(ply + 1) / 2}, ${movePly === ply ? 'White' : 'Black'} ${notation[movePly - 1]}`}
                  onclick={() => onreview(movePly)}
                  onkeydown={(event) => navigate(event, movePly)}
                >
                  {notation[movePly - 1]}
                </button>
              {:else}
                <span class="note">—</span>
              {/if}
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
