import unittest
from dataclasses import replace
import numpy as np
import torch
from future_vr.config import make_config
from major_revision.action_value import action_size, observation_size, prior_values, action_episode
from major_revision.gram_control import representation_size
from major_revision.horizon_prior import block_costs, horizon_values, HorizonDQN, FixedHorizon
from major_revision.horizon_prior_screen import gate
from major_revision.pilot import trajectory


class HorizonPriorTests(unittest.TestCase):
    def test_cost_units_and_discount_are_per_four_slot_block(self):
        cfg = make_config('validation')
        state = np.zeros(observation_size(cfg), dtype=np.float32)
        rows = state[representation_size(cfg):].reshape(cfg.subarrays, action_size(cfg))
        rows[:, 16:20] = .25
        rows[:, 20:24] = .5
        rows[:, 24:48] = 2.
        rows[:, 24:27] = .5  # Three outages out of 12 user-slots in block zero.
        rows[:, 36:42] = .5  # Six out of 12 in block one.
        costs = block_costs(cfg, torch.from_numpy(state))
        np.testing.assert_allclose(costs[:, 0], .75)
        np.testing.assert_allclose(costs[:, 1], 1.5)
        np.testing.assert_allclose(-costs[:, 0], prior_values(cfg, state))
        np.testing.assert_allclose(horizon_values(cfg, torch.from_numpy(state)), -.75 - cfg.rl_gamma * 1.5)
        with self.assertRaises(ValueError):
            block_costs(replace(cfg, horizon=4), torch.from_numpy(state))

    def test_initial_network_equals_fixed_prior_and_training_has_successor(self):
        cfg = replace(make_config('validation'), episode_steps=12)
        tr = trajectory(replace(cfg, episode_steps=13), 995900240, (2., 1.))
        fixed = action_episode(cfg, tr, FixedHorizon(cfg))
        model = HorizonDQN(cfg, 11)
        self.assertEqual(sum(p.numel() for p in model.online.parameters()), 31873)
        initial = action_episode(cfg, tr, model)
        for key in ('energy_j', 'outage', 'mean_count'):
            self.assertAlmostEqual(initial[key], fixed[key], places=9)
        action_episode(cfg, tr, model, training=True)
        self.assertEqual(len(model.memory), 3)
        self.assertTrue(all(not item.done for item in model.memory))
        item = model.memory[-1]
        state = torch.as_tensor(item.state).unsqueeze(0)
        torch.testing.assert_close(model.online(state)[0], horizon_values(cfg, state)[0])
        for _ in range(65):
            model.remember(item)
        before = model.online.head[-1].weight.detach().clone()
        self.assertTrue(np.isfinite(model.learn()))
        self.assertFalse(torch.equal(before, model.online.head[-1].weight))

    def test_gate_checks_all_scenarios_and_requires_two_feasible(self):
        previous = [dict(outage=x, energy_j=100.) for x in (.08, .33, .54)]
        full = [dict(outage=0., energy_j=300.) for _ in range(3)]
        candidate = [dict(outage=x, energy_j=120.) for x in (.02, .04, .2)]
        self.assertTrue(gate(candidate, previous, full)['permit_training'])
        candidate[1]['outage'] = .06
        self.assertFalse(gate(candidate, previous, full)['permit_training'])
        candidate[1]['outage'] = .04
        candidate[2]['energy_j'] = 301.
        self.assertFalse(gate(candidate, previous, full)['permit_training'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
