import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pgx.experimental.chess import from_fen

from bridge import ACTIONS, decode_moves, encode, make_recurrent, make_search, replay


def uniform(_, state):
    return jnp.zeros(ACTIONS), jnp.float32(0.25)


def batch(*states):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)


class BridgeTest(unittest.TestCase):
    def test_action_conversion(self):
        initial = replay([])
        self.assertEqual(len(decode_moves(initial)), 20)
        self.assertIn("e2e4", decode_moves(initial).values())
        with self.assertRaisesRegex(ValueError, "Illegal move"):
            replay(["e2e5"])
        castle = from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        self.assertTrue({"e8g8", "e8c8"}.issubset(decode_moves(castle).values()))
        promotion = from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")
        self.assertTrue(
            {f"a7a8{piece}" for piece in "qrbn"}.issubset(
                decode_moves(promotion).values()
            )
        )

    def test_encoding_and_en_passant(self):
        initial = np.asarray(encode(replay([])))
        self.assertEqual(initial.shape, (782,))
        self.assertEqual(initial[48], 1)  # Own pawn on a2.
        self.assertEqual(initial[384 + 8], 1)  # Opposing pawn on a7.
        after_e4 = np.asarray(encode(replay(["e2e4"])))
        self.assertEqual(after_e4[-1], 1)
        self.assertEqual(after_e4[772:780].sum(), 0)
        ep = replay(["e2e4", "a7a6", "e4e5", "d7d5"])
        self.assertIn("e5d6", decode_moves(ep).values())
        self.assertEqual(float(encode(ep)[775]), 1)

    def test_rewards_and_terminal_absorption(self):
        mate = replay(["f2f3", "e7e5", "g2g4"])
        quiet = replay([])
        states = batch(mate, quiet)
        checkmate = next(a for a, move in decode_moves(mate).items() if move == "d8h4")
        e4 = next(a for a, move in decode_moves(quiet).items() if move == "e2e4")
        recurrent = jax.jit(make_recurrent(uniform))
        output, next_states = recurrent(
            None, jax.random.key(0), jnp.array([checkmate, e4]), states
        )
        np.testing.assert_array_equal(output.reward, [1, 0])
        np.testing.assert_array_equal(output.discount, [0, -1])
        np.testing.assert_array_equal(output.value, [0, 0.25])
        np.testing.assert_array_equal(next_states.terminated, [True, False])
        self.assertNotEqual(int(states.current_player[1]), int(next_states.current_player[1]))
        absorbed, _ = recurrent(
            None, jax.random.key(1), jnp.array([0, 0]), next_states
        )
        self.assertEqual(float(absorbed.reward[0]), 0)
        self.assertEqual(float(absorbed.discount[0]), 0)
        draw = replay(["g1f3", "g8f6", "f3g1", "f6g8"] * 2)
        self.assertTrue(bool(draw.terminated))
        np.testing.assert_array_equal(draw.rewards, [0, 0])

    def test_real_mctx_returns_legal_actions(self):
        roots = [replay([]), replay(["e2e4"])]
        output = make_search(uniform, simulations=8, max_depth=3)(
            None, jax.random.key(3), batch(*roots)
        )
        for root, action, policy in zip(roots, output.action, output.action_weights):
            self.assertIn(int(action), decode_moves(root))
            self.assertAlmostEqual(float(policy.sum()), 1, places=5)
            self.assertTrue(np.all(np.asarray(policy)[~root.legal_action_mask] == 0))


if __name__ == "__main__":
    unittest.main()
