<script lang="ts">
import type { ModelHistoryEntry } from './modelHistory';

let { active = true }: { active?: boolean } = $props();
let entries = $state<ModelHistoryEntry[]>([]);
let requested = $state(false);
let status = $state('Loading model history…');
const date = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
});

async function load() {
  const url = new URL(
    `${import.meta.env.BASE_URL}model-history.jsonl`,
    location.href,
  );
  try {
    const response = await fetch(url);
    if (response.status === 404) {
      status = 'Results appear after the first Stockfish benchmark.';
      return;
    }
    if (!response.ok) throw new Error('Could not load model history.');
    entries = (await response.text())
      .trim()
      .split('\n')
      .filter(Boolean)
      .slice(-8)
      .reverse()
      .map((line) => JSON.parse(line));
    status = entries.length
      ? ''
      : 'Results appear after the first Stockfish benchmark.';
  } catch {
    status = 'Model history is temporarily unavailable.';
  }
}

$effect(() => {
  if (active && !requested) {
    requested = true;
    void load();
  }
});
</script>

<h2>Model history</h2>
{#if status}
  <p class="note" role="status">{status}</p>
{:else}
  <p>
    Average expected-score points lost against Stockfish on
    {entries[0].positionCount} fixed positions. Lower is better.
  </p>
  <p class="note">
    Latest held-out MAE · incumbent
    {entries[0].incumbentMae?.toFixed(3) ?? '—'} · candidate
    {entries[0].candidateMae.toFixed(3)}
  </p>
  <table aria-label="Recent training runs">
    <thead>
      <tr>
        <th scope="col">Run</th>
        <th scope="col">Incumbent</th>
        <th scope="col">Candidate</th>
      </tr>
    </thead>
    <tbody>
      {#each entries as entry (entry.runId)}
        <tr>
          <th scope="row">
            <a
              href={`https://github.com/RajeevAtla/choux/actions/runs/${entry.runId}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <time datetime={entry.timestamp}
                >{date.format(new Date(entry.timestamp))}</time
              >
            </a>
            <small class="note"
              >{entry.promoted ? 'Promoted' : 'Kept incumbent'}</small
            >
          </th>
          <td>{entry.incumbent.meanLoss.toFixed(2)}</td>
          <td>{entry.candidate.meanLoss.toFixed(2)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}
