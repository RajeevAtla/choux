import type { Chess, Move } from 'chess.js';
import { moveBudgetMs, rankMoves } from './engine';

type Evaluate = Parameters<typeof rankMoves>[1];
type Edge = {
  move: Move;
  estimate: number;
  terminal: boolean;
  prior: number;
  visits: number;
  total: number;
  children?: Edge[];
};

const maxRootActions = 8;
const maxDepth = 4;

function mean(edge: Edge) {
  return edge.visits ? edge.total / edge.visits : edge.estimate;
}

function improvedLogit(edge: Edge) {
  return edge.prior + 2 * mean(edge);
}

async function edges(
  game: Chess,
  evaluate: Evaluate,
  deadline: number,
): Promise<Edge[]> {
  const ranked = await rankMoves(game, evaluate, deadline);
  return ranked.map(({ move, score, terminal }) => {
    const estimate = Math.min(1, score);
    return {
      move,
      estimate,
      terminal,
      prior: estimate + Math.tanh(evaluate.movePrior?.(game, move) ?? 0),
      visits: 0,
      total: 0,
    };
  });
}

// Follow the improved policy while accounting for simulations already spent.
function nextEdge(children: Edge[]) {
  const logits = children.map(improvedLogit);
  const max = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - max));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const visits = children.reduce((sum, child) => sum + child.visits, 0);
  let best = -Infinity;
  let chosen = children[0];
  for (const [index, child] of children.entries()) {
    const priority = weights[index] / totalWeight - child.visits / (visits + 1);
    if (priority > best) {
      best = priority;
      chosen = child;
    }
  }
  return chosen;
}

// Adapt Gumbel MuZero's root sequential halving for deterministic browser play.
export async function chooseMoveMctx(
  game: Chess,
  evaluate: Evaluate,
  budgetMs = moveBudgetMs,
) {
  if (game.isGameOver()) return undefined;
  const deadline = performance.now() + budgetMs;
  const legal = game.moves({ verbose: true });
  const mate = legal.find((move) => move.san.endsWith('#'));
  if (mate || legal.length <= 1) return mate ?? legal[0];
  if (performance.now() >= deadline) return legal[0];

  let root = (await edges(game, evaluate, deadline))
    .sort((a, b) => b.prior - a.prior)
    .slice(0, maxRootActions);
  if (!root.length) return legal[0];
  let best = root[0];

  async function simulate(
    edge: Edge,
    depth: number,
  ): Promise<number | undefined> {
    if (performance.now() >= deadline) return undefined;
    let value = edge.estimate;
    if (!edge.terminal && depth > 0) {
      game.move(edge.move);
      try {
        if (edge.children?.length) {
          const response = await simulate(nextEdge(edge.children), depth - 1);
          if (response === undefined) return undefined;
          value = -response;
        } else {
          edge.children = await edges(game, evaluate, deadline);
          if (performance.now() >= deadline) return undefined;
          if (edge.children.length) value = -edge.children[0].estimate;
        }
      } finally {
        game.undo();
      }
    }
    edge.visits++;
    edge.total += value;
    return value;
  }

  for (let round = 0; root.length > 1; round++) {
    const target = 2 ** round;
    for (let visit = 0; visit < target; visit++) {
      for (const edge of root) {
        if ((await simulate(edge, maxDepth)) === undefined) return best.move;
      }
      // Equal simulation counts let incomplete rounds use the last fair pass.
      best = root.reduce((leader, edge) =>
        improvedLogit(edge) > improvedLogit(leader) ? edge : leader,
      );
    }
    root = root
      .sort((a, b) => improvedLogit(b) - improvedLogit(a))
      .slice(0, Math.ceil(root.length / 2));
  }
  return best.move;
}
