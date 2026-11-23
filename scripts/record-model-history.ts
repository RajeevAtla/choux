import type { ModelHistoryEntry } from '../src/modelHistory';

const [trainingPath, stockfishPath, promoted] = Bun.argv.slice(2);
const runId = Bun.env.GITHUB_RUN_ID;
if (
  !runId ||
  !trainingPath ||
  !stockfishPath ||
  !['true', 'false'].includes(promoted)
) {
  throw new Error(
    'Usage: GITHUB_RUN_ID=<id> bun scripts/record-model-history.ts <training report> <Stockfish report> <promoted>',
  );
}

const path = 'public/model-history.jsonl';
const history = Bun.file(path);
const rows: ModelHistoryEntry[] = (await history.exists())
  ? (await history.text())
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line))
  : [];
const { eligible, candidateMae, incumbentMae, trainingPositions } =
  await Bun.file(trainingPath).json();
const { suite, engine, nodes, positionCount, candidate, incumbent } =
  await Bun.file(stockfishPath).json();
const entry: ModelHistoryEntry = {
  runId,
  timestamp: new Date().toISOString(),
  promoted: promoted === 'true',
  eligible,
  candidateMae,
  incumbentMae,
  trainingPositions,
  suite,
  engine,
  nodes,
  positionCount,
  candidate,
  incumbent,
};

const retained = rows.filter((row) => row.runId !== runId);
await Bun.write(
  path,
  `${[...retained, entry].map((row) => JSON.stringify(row)).join('\n')}\n`,
);
