import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.risk_budget.control import action_at, choose, episode
from joint_power.study import make_trace


class BudgetTests(unittest.TestCase):
    def test_feasibility_then_energy_not_risk_credit(self):
        action, count = choose([0., -2., -.5], [.1, 2., .4], .5, 0)
        self.assertEqual((action, count), (2, 2))
        self.assertEqual(choose([0., 1.], [.4, .1], .5, 0)[0], 0)

    def test_empty_feasible_set_and_reference_ties(self):
        self.assertEqual(choose([0., -1., 2.], [2., 2., 1.], .1, 0), (2, 0))
        self.assertEqual(choose(np.zeros(5), np.ones(5), 2., 3), (3, 5))
        self.assertEqual(choose(np.zeros(5), np.ones(5), 0., 3), (3, 0))
        with self.assertRaises(ValueError):
            choose([0.], [-1.], 0., 0)

    def test_future_channel_does_not_change_current_command(self):
        torch.set_num_threads(1)
        cfg = replace(make_config('validation'), episode_steps=20)
        trace = make_trace(cfg, 4299220000, (2., 1.))
        env = Environment(cfg, trace)
        env.block(env.candidates().masks[2])
        changed = copy.deepcopy(env)
        channel, gain = trace.channel.copy(), trace.gain_db.copy()
        channel[env.t+1:] *= .01
        gain[env.t+1:] -= 40
        changed.tr = replace(trace, channel=channel, gain_db=gain)
        loaded, _ = models()
        model, predictor = loaded['Visitation-added', 11], load_predictor()
        for rule in ('Value', 'Myopic', 'Raw budget', 'PRS budget'):
            command, info = action_at(env, predictor, model, rule, .5, 1)
            altered, other = action_at(changed, predictor, model, rule, .5, 1)
            np.testing.assert_array_equal(command, altered)
            self.assertEqual(info, other)

    def test_ledger_matches_executed_failures(self):
        torch.set_num_threads(1)
        cfg = replace(make_config('validation'), episode_steps=20)
        trace = make_trace(cfg, 4299220001, (3., 1.15))
        _, records, decisions = episode(trace, 'Raw budget', 1., load_predictor(), cfg=cfg)
        past = 0
        for i, row in enumerate(decisions):
            self.assertEqual(row['past_failed'], past)
            self.assertAlmostEqual(row['residual_budget'], max(0., .05*3*20-past))
            failed = sum(round(3*r['outage']) for r in records[4*i:4*i+4])
            self.assertEqual(row['block_failed'], failed)
            past += failed
            self.assertEqual(row['cumulative_failed'], past)


if __name__ == '__main__':
    unittest.main()
