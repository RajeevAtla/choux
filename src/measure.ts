import {
  type Backend,
  type BenchmarkRequest,
  type BenchmarkResult,
  TRIALS,
  WORKLOADS,
} from './benchmark';

function request(
  worker: Worker,
  message: BenchmarkRequest,
): Promise<BenchmarkResult> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(
      () => reject(new Error('The worker timed out.')),
      20_000,
    );
    worker.onmessage = ({ data }: MessageEvent<BenchmarkResult>) => {
      window.clearTimeout(timeout);
      resolve(data);
    };
    worker.onerror = (error) => {
      window.clearTimeout(timeout);
      reject(new Error(error.message || 'The worker could not start.'));
    };
    worker.postMessage(message);
  });
}

export async function measure(backend: Backend) {
  const checkpointUrl = new URL(
    `${import.meta.env.BASE_URL}benchmark-weights.f32`,
    location.href,
  ).href;
  const start = performance.now();
  const worker = new Worker(new URL('./benchmark.worker.ts', import.meta.url), {
    type: 'module',
  });
  try {
    const ready = await request(worker, { backend, checkpointUrl });
    const coldMs = performance.now() - start;
    if (ready.status === 'unavailable') return ready;
    if (ready.status !== 'ready')
      throw new Error(
        'message' in ready ? ready.message : 'Invalid startup response.',
      );

    const rows = [];
    for (const workload of WORKLOADS) {
      const times = [];
      for (let trial = 0; trial < TRIALS; trial++) {
        const start = performance.now();
        const result = await request(worker, { workload, trial });
        const elapsed = performance.now() - start;
        if (
          result.status !== 'measured' ||
          result.scores.length !== workload ||
          !result.scores.every(Number.isFinite)
        ) {
          throw new Error(
            'message' in result ? result.message : 'Invalid inference scores.',
          );
        }
        times.push(elapsed);
      }

      times.sort((a, b) => a - b);
      const middle = Math.floor(times.length / 2);
      rows.push({
        workload,
        median: (times[middle - 1] + times[middle]) / 2,
        p95: times[Math.ceil(times.length * 0.95) - 1],
      });
    }

    return { ...ready, coldMs, rows };
  } finally {
    worker.terminate();
  }
}
