"""Policy-port equivalence and information access using synthetic fixtures."""
import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import action_at
from joint_power.model import Environment
from joint_power.standard_transfer.control import corrected_search, episode, learned_action
from joint_power.standard_transfer.environment import WidebandEnvironment
from joint_power.study import make_trace


class TransferControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(make_config('validation'), episode_steps=12)
        cls.trace = make_trace(cls.cfg, 5899220002, (2., 1.))
        cls.flat = np.repeat(cls.trace.channel[:, None], 3, axis=1)
        cls.model = models()[0]['Visitation-added', 11]
        cls.predictor = load_predictor()

    def test_flat_channel_port_matches_frozen_action_and_ledger(self):
        old, new = Environment(self.cfg, self.trace), WidebandEnvironment(self.cfg, self.trace, self.flat)
        past = 0
        while old.t < self.cfg.episode_steps:
            command, info = action_at(old, self.predictor, self.model, 'Value', .1, past, 4)
            new_command, new_info = learned_action(new, self.predictor, self.model, past)
            np.testing.assert_array_equal(command, new_command)
            self.assertEqual(info, new_info)
            start = len(old.records)
            old.block(command)
            new.block(new_command)
            past += sum(round(3*r['outage']) for r in old.records[start:])

    def test_hidden_current_and_future_csi_do_not_change_decisions(self):
        env = WidebandEnvironment(self.cfg, self.trace, self.flat.copy())
        env.block(np.arange(16) < 2)
        other = copy.deepcopy(env)
        other.physical[env.t+1:] *= 1000
        other.physical[env.t][:, :, np.repeat(~env.hw.active, 8)] *= 1000
        for minimum in (1, 4):
            np.testing.assert_array_equal(corrected_search(env, .05, minimum),
                corrected_search(other, .05, minimum))
        a, info = learned_action(env, self.predictor, self.model, 0)
        b, other_info = learned_action(other, self.predictor, self.model, 0)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(info, other_info)

    def test_all_references_complete_with_matching_domains_and_counts(self):
        for method in ('PRS4', 'budgetPRS4', 'PRS1', 'budgetPRS1', 'WBbudgetPRS4', 'WBbudgetPRS1', 'Full array'):
            _, records, ledger = episode(self.cfg, self.trace, self.flat, method)
            self.assertEqual(len(records), 12)
            minimum = 16 if method == 'Full array' else int(method[-1])
            self.assertTrue(all(r['command_bits'].bit_count() >= minimum for r in records))
            self.assertEqual(ledger[-1]['cumulative_failed'], sum(round(3*r['outage']) for r in records))


if __name__ == '__main__':
    unittest.main()
