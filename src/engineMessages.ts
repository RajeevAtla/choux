export type EngineRequest = { id: number; checkpoint: string } & (
  | { type: 'load' }
  | { type: 'move'; moves: string[]; search?: 'tactical' | 'mctx' }
);

export type EngineResponse = { id: number } & (
  | { type: 'ready'; model: string }
  | { type: 'move'; move: string }
  | { type: 'error'; message: string }
);
