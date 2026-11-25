export type ModelScore = {
  hash: string;
  meanLoss: number;
  bestMoveMatches: number;
};

export type ModelHistoryEntry = {
  runId: string;
  timestamp: string;
  promoted: boolean;
  eligible: boolean;
  candidateMae: number;
  validationSource?: 'stockfish-v1' | 'heuristic';
  incumbentMae: number | null;
  trainingPositions: number;
  suite: string;
  engine: string;
  nodes: number;
  positionCount: number;
  candidate: ModelScore;
  incumbent: ModelScore;
};
