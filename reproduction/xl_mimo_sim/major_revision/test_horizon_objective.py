import unittest
import numpy as np
from future_vr.config import make_config
from major_revision.action_value import observation_size, action_size
from major_revision.gram_control import representation_size
from major_revision.horizon_objective_audit import inspect_choice


class HorizonObjectiveTests(unittest.TestCase):
    def test_price_can_prefer_risky_action_over_available_feasible_one(self):
        cfg = make_config('validation')
        state = np.zeros(observation_size(cfg), dtype=np.float32)
        rows = state[representation_size(cfg):].reshape(cfg.subarrays, action_size(cfg))
        rows[:, 16:24] = .9
        rows[:, 24:48] = 2.
        rows[0, 16:24] = .1
        rows[0, 24:27] = .5
        rows[0, 36:39] = .5
        chosen, info = inspect_choice(cfg, state)
        self.assertEqual(chosen, 0)
        self.assertEqual(info['category'], 'unsafe_with_feasible_alternative')
        self.assertEqual(info['feasible_actions'], 15)
        self.assertGreater(info['chosen_q'], info['alternative_q'])
        rows[0, 24:48] = 2.
        self.assertEqual(inspect_choice(cfg, state)[1]['category'], 'chosen_predicted_feasible')
        rows[:, 24:48] = .5
        self.assertEqual(inspect_choice(cfg, state)[1]['category'], 'no_predicted_feasible_action')


if __name__ == '__main__':
    unittest.main()
