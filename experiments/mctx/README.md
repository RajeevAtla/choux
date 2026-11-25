# Offline mctx search

The Python experiment runs upstream [mctx](https://github.com/google-deepmind/mctx)
with [Pgx chess](https://www.sotets.uk/pgx/chess/) as the recurrent transition.
Pgx supplies 4,672 action IDs, the legal-action mask, and the next game state.
Nonterminal edges use a discount of −1 to switch value perspective. A terminal
edge pays the chess result to the mover once and has a discount of zero.

From the repository root, with Python 3.13 and uv:

```sh
uv run --no-project --python 3.13 --with-requirements experiments/mctx/requirements.txt python experiments/mctx/bridge.py --simulations 32 e2e4 e7e5 g1f3
uv run --no-project --python 3.13 --with-requirements experiments/mctx/requirements.txt python -m unittest discover -s experiments/mctx -v
```

The arguments after the options are UCI moves from the initial board. Replaying
the history preserves repetition and castling state. With no moves, search starts
from the initial board. The command prints a JSON object containing a UCI move,
its Pgx action ID, and the improved distribution over legal moves. Use
`--checkpoint` to select a different Choux checkpoint. The offline experiment
uses the value network and uniform legal move priors.

Browser play offers `?search=mctx` as an opt-in mode. That implementation adapts
sequential halving and visit-based tree selection to chess.js transitions and
JAX-JS evaluation. It uses a soft 200 ms budget instead of a fixed simulation
count, and also uses the learned move prior when the checkpoint has one.
