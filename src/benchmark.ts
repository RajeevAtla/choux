export const BATCH_SIZE = 64;
export const WORKLOADS = [64, 512] as const;
export const TRIALS = 30;
export const BACKENDS = ['wasm', 'webgpu'] as const;
export type Backend = (typeof BACKENDS)[number];

export type BenchmarkResult =
  | { status: 'ready'; setupMs: number; firstBatchMs: number; maxError: number }
  | { status: 'unavailable' | 'error'; message: string }
  | { status: 'measured'; scores: Float32Array };

export type BenchmarkRequest =
  | { backend: Backend; checkpointUrl: string }
  | { workload: number; trial: number };
