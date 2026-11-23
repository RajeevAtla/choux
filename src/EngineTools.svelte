<script lang="ts">
import { BACKENDS, type Backend, TRIALS } from './benchmark';
import ModelHistory from './ModelHistory.svelte';
import { measure } from './measure';
import { HIDDEN, INPUTS, WEIGHT_COUNT } from './modelShape';

let {
  checkpoint = '',
  active = true,
}: { checkpoint?: string; active?: boolean } = $props();

const names = { wasm: 'Wasm', webgpu: 'WebGPU' };
let running = $state(false);
let status = $state('Ready to measure.');
let results = $state<
  Partial<
    Record<Backend, Awaited<ReturnType<typeof measure>> | { message: string }>
  >
>({});

async function run() {
  running = true;
  results = {};
  let completed = 0;
  for (const backend of BACKENDS) {
    status = `Measuring ${names[backend]}…`;
    try {
      const result = await measure(backend);
      results[backend] = result;
      if ('rows' in result) completed++;
    } catch (error) {
      results[backend] = {
        message: error instanceof Error ? error.message : String(error),
      };
    }
  }
  status = completed
    ? 'Benchmark complete.'
    : 'No backend completed the benchmark.';
  running = false;
}
</script>

<section class="engine-tools" aria-label="Engine and model tools">
  <h2>Model</h2>
  <p>
    Play runs locally with JAX-JS on Wasm. The network has {INPUTS} inputs,
    {HIDDEN} hidden units, one output, and
    {WEIGHT_COUNT.toLocaleString('en-US')} parameters.
  </p>
  {#if checkpoint}
    <p class="note">
      Current checkpoint
      <code title={checkpoint}>{checkpoint.slice(0, 12)}</code>
    </p>
  {/if}
  <ModelHistory {active} />
  <h2>Inference benchmark</h2>
  <p>Compare Wasm and WebGPU using a dummy chess model in a browser worker.</p>
  <button type="button" onclick={run} disabled={running}>Run benchmark</button>
  <p id="status" role="status">{status}</p>
  <section aria-label="Benchmark results">
    {#each BACKENDS as backend (backend)}
      {@const result = results[backend]}
      {#if result}
        <article>
          <h3>{names[backend]}</h3>
          {#if 'rows' in result}
            <p>
              Cold start {result.coldMs.toFixed(1)} ms · setup
              {result.setupMs.toFixed(1)} ms · first batch
              {result.firstBatchMs.toFixed(1)} ms · max score error
              {result.maxError.toExponential(1)}
            </p>
            <table aria-label={`${names[backend]} warm timings`}>
              <thead>
                <tr>
                  <th scope="col">Positions</th>
                  <th scope="col">Median (ms)</th>
                  <th scope="col">P95 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {#each result.rows as row (row.workload)}
                  <tr>
                    <td>{row.workload}</td>
                    <td>{row.median.toFixed(2)}</td>
                    <td>{row.p95.toFixed(2)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {:else}
            <p>{result.message}</p>
          {/if}
        </article>
      {/if}
    {/each}
  </section>
  <p class="note">
    Warm timings cover feature generation, inference, score readback, and the
    worker round trip. Each workload runs {TRIALS} times.
  </p>
</section>
