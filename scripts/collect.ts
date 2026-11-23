import { submissionTitle } from '../src/submission';
import { datasetPath, parseSubmission, readGames } from './dataset';

type Issue = { title: string; body: string | null; pull_request?: unknown };
const maxNewGames = 100;
const token = Bun.env.GITHUB_TOKEN;
const repo = Bun.env.GITHUB_REPOSITORY;
if (!token || !repo)
  throw new Error('GITHUB_TOKEN and GITHUB_REPOSITORY are required.');

const dataset = await readGames();
const seen = new Set(dataset.map(({ moves }) => moves.join(' ')));
let added = 0;

for (let page = 1; added < maxNewGames; page++) {
  const endpoint = `https://api.github.com/repos/${repo}/issues?state=all&sort=created&direction=asc&per_page=100&page=${page}`;
  const response = await fetch(endpoint, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
    },
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok)
    throw new Error(
      `GitHub issue request failed with status ${response.status}.`,
    );

  const issues: Issue[] = await response.json();
  for (const issue of issues) {
    if (issue.pull_request || issue.title !== submissionTitle || !issue.body)
      continue;
    const game = parseSubmission(issue.body);
    if (!game) continue;
    const key = game.moves.join(' ');
    if (seen.has(key)) continue;
    seen.add(key);
    dataset.push(game);
    if (++added === maxNewGames) break;
  }
  if (issues.length < 100) break;
}

if (added)
  await Bun.write(
    datasetPath,
    `${dataset.map((game) => JSON.stringify(game)).join('\n')}\n`,
  );
console.log(
  `Accepted ${added} new games. The dataset contains ${dataset.length} games.`,
);
