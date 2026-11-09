import unittest
from dataclasses import replace
import numpy as np
import torch

from future_vr.config import make_config
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.risk_correction import select_action
from major_revision.risk_rl import RiskDQN, state, episode
from major_revision.risk_rl_study import setup


class MatchedRiskRLTests(unittest.TestCase):
    def test_zero_network_search_parity_masks_and_hidden_truth(self):
        torch.set_num_threads(1)
        cfg = make_config('validation')
        settings, risk = setup()
        for mode in ('old', 'calibrated'):
            controller = RiskDQN(cfg, 11)
            run = FullDomainRollout(cfg, trajectory(cfg, 1190900001, (2., 1.)))
            while run.t < 40:
                if run.t % 4:
                    run.step()
                    continue
                observation, data, allowed = state(run, mode, *settings[mode, 1], risk)
                self.assertEqual(len(observation), 1658)
                action, _ = controller.choose(observation, 0.)
                self.assertEqual(action, select_action(data, run.req, mode, *settings[mode, 1], risk))
                for _ in range(10):
                    exploratory, _ = controller.choose(observation, 1.)
                    self.assertTrue(allowed[exploratory])
                q = controller.online(torch.as_tensor(observation)[None]).detach().numpy()[0]
                self.assertTrue(np.all(q[~allowed] == -1e9))
                changed = run.clone()
                h = run.tr.channel.copy()
                h[run.t+1:] *= 1e8
                changed.tr = replace(run.tr, channel=h)
                other, _, _ = state(changed, mode, *settings[mode, 1], risk)
                np.testing.assert_array_equal(observation, other)
                run.step(data['masks'][action])

    def test_training_has_real_successor_and_nonterminal_records(self):
        cfg = replace(make_config('validation'), episode_steps=12)
        settings, risk = setup()
        controller = RiskDQN(cfg, 22)
        tr = trajectory(replace(cfg, episode_steps=13), 1190900002, (2., 1.))
        result = episode(cfg, tr, 'calibrated', *settings['calibrated', 1], risk, controller, True, .2)
        self.assertEqual(len(controller.memory), 3)
        self.assertTrue(all(not t.done for t in controller.memory))
        self.assertTrue(all(np.isfinite(t.next_state).all() for t in controller.memory))
        self.assertGreater(result['energy_j'], 0)
        with self.assertRaises(ValueError):
            episode(cfg, replace(tr, channel=tr.channel[:12]), 'calibrated',
                    *settings['calibrated', 1], risk, controller, True)


if __name__ == '__main__':
    unittest.main()
