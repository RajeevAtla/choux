"""Exact chess transitions and the Choux value model for Python mctx."""

import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from pgx.chess import Chess
from pgx._src.games.chess import FROM_PLANE, PAWN, TO_PLANE

ENV = Chess()
ACTIONS = ENV.num_actions
INPUTS = 782
HIDDEN = 64
VALUE_WEIGHTS = INPUTS * HIDDEN + 2 * HIDDEN + 1
POLICY_WEIGHTS = 785
STEP = jax.jit(ENV.step)
INIT = jax.jit(ENV.init)


def encode(state):
    """Match src/position.ts, including side-relative planes and castling."""
    board = state._x
    rank_major = jnp.rot90(board.board.reshape(8, 8)).reshape(64)
    pieces = jnp.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
    planes = (rank_major[None, :] == pieces[:, None]).reshape(-1)
    castling = board.castling_rights[:, ::-1].reshape(-1)
    # chess.js includes the en passant square only when a legal capture exists.
    ep = jnp.maximum(board.en_passant, 0)
    sources = ep - 1 + jnp.array([-8, 8])
    valid = (sources >= 0) & (sources < 64)
    indices = sources.clip(0, 63)
    plane = TO_PLANE[indices, ep]
    captures = valid & (plane >= 0) & (board.board[indices] == PAWN)
    legal_ep = (board.en_passant >= 0) & (
        captures & state.legal_action_mask[indices * 73 + plane.clip(0, 72)]
    ).any()
    ep_files = jax.nn.one_hot(ep // 8, 8) * legal_ep
    clocks = jnp.array([jnp.minimum(board.halfmove_count, 100) / 100, board.color])
    return jnp.concatenate((planes, castling, ep_files, clocks))


def load_checkpoint(path):
    data = np.fromfile(path, dtype="<f4")
    if data.size not in (VALUE_WEIGHTS, VALUE_WEIGHTS + POLICY_WEIGHTS):
        raise ValueError("Unexpected Choux checkpoint size")
    if not np.isfinite(data).all():
        raise ValueError("Checkpoint contains invalid weights")
    split = INPUTS * HIDDEN
    return (
        jnp.asarray(data[:split].reshape(INPUTS, HIDDEN)),
        jnp.asarray(data[split:split + HIDDEN]),
        jnp.asarray(data[split + HIDDEN:VALUE_WEIGHTS - 1]),
        jnp.asarray(data[VALUE_WEIGHTS - 1]),
    )


def choux_model(weights, state):
    inputs, hidden_bias, output, output_bias = weights
    value = jax.nn.relu(encode(state) @ inputs + hidden_bias) @ output + output_bias
    return jnp.zeros(ACTIONS, dtype=jnp.float32), jnp.clip(value, -1, 1)


def live(state):
    return ~(state.terminated | state.truncated)


def evaluate(model, params, state):
    logits, value = model(params, state)
    # Absorbing terminal nodes get a dummy action with no reward and no value.
    terminal_mask = jnp.arange(ACTIONS) == 0
    mask = jnp.where(live(state), state.legal_action_mask, terminal_mask)
    return jnp.where(mask, logits, jnp.finfo(logits.dtype).min), jnp.where(
        live(state), value, 0.0
    )


def make_recurrent(model):
    @jax.vmap
    def transition(key, action, state):
        legal = live(state) & state.legal_action_mask[action]
        next_state = jax.lax.cond(
            legal, lambda: ENV.step(state, action, key), lambda: state
        )
        reward = jnp.where(legal, next_state.rewards[state.current_player], 0.0)
        # PGX and the value model both change perspective on each legal ply.
        discount = jnp.where(legal & live(next_state), -1.0, 0.0)
        return reward, discount, next_state

    def recurrent(params, key, actions, states):
        keys = jax.random.split(key, actions.shape[0])
        rewards, discounts, next_states = transition(keys, actions, states)
        logits, values = jax.vmap(evaluate, in_axes=(None, None, 0))(
            model, params, next_states
        )
        output = mctx.RecurrentFnOutput(
            reward=rewards, discount=discounts, prior_logits=logits, value=values
        )
        return output, next_states

    return recurrent


def make_search(model=choux_model, simulations=32, max_depth=6):
    """Return a compiled batched search. Each root must be a live position."""
    if simulations < 2 or max_depth < 1:
        raise ValueError("Require at least two simulations and positive depth")
    recurrent = make_recurrent(model)

    @jax.jit
    def search(params, key, states):
        logits, values = jax.vmap(evaluate, in_axes=(None, None, 0))(
            model, params, states
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=values, embedding=states)
        return mctx.gumbel_muzero_policy(
            params, key, root, recurrent, simulations,
            invalid_actions=~states.legal_action_mask,
            max_depth=max_depth,
            max_num_considered_actions=min(16, simulations),
            gumbel_scale=0.0,
        )

    return search


def decode_moves(state):
    """Map legal Pgx action IDs to UCI moves on the host."""
    black = int(state._x.color)
    board = np.asarray(state._x.board)
    destinations = np.asarray(FROM_PLANE)

    def square(index):
        return chr(97 + index // 8) + str(1 + ((index % 8) ^ (7 * black)))

    moves = {}
    if not bool(live(state)):
        return moves
    for action in np.flatnonzero(np.asarray(state.legal_action_mask)):
        source, plane = divmod(int(action), 73)
        target = int(destinations[source, plane])
        promotion = "rbn"[plane // 3] if plane < 9 else ""
        if not promotion and board[source] == PAWN and target % 8 == 7:
            promotion = "q"
        moves[int(action)] = square(source) + square(target) + promotion
    return moves


def replay(moves):
    state = INIT(jax.random.key(0))
    for move in moves:
        legal = {uci: action for action, uci in decode_moves(state).items()}
        if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) or move not in legal:
            raise ValueError(f"Illegal move: {move}")
        state = STEP(state, jnp.int32(legal[move]))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moves", nargs="*", help="UCI move history from the initial board")
    parser.add_argument("--checkpoint", type=Path, default=Path("public/model.f32"))
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()
    state = replay(args.moves)
    legal = decode_moves(state)
    if not legal:
        parser.error("The position is terminal")
    batch = jax.tree.map(lambda leaf: leaf[None], state)
    policy = make_search(simulations=args.simulations)(
        load_checkpoint(args.checkpoint), jax.random.key(1), batch
    )
    action = int(policy.action[0])
    weights = np.asarray(policy.action_weights[0])
    print(json.dumps({
        "move": legal[action],
        "action": action,
        "policy": {move: float(weights[index]) for index, move in legal.items()},
    }))


if __name__ == "__main__":
    main()
