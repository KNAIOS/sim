import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.baseline_anchor.control import episode as original_episode
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import allowed, episode, raw_action, restricted_choice
from joint_power.model import Environment, search_action
from joint_power.study import make_trace


class MatchedDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(make_config('validation'), episode_steps=20)
        cls.trace = make_trace(cls.cfg, 5199220004, (2., 1.))
        cls.predictor = load_predictor()
        cls.model = models()[0]['Visitation-added', 11]

    def test_minimum_one_is_frozen_implementation(self):
        for proposer, model in (('Value', self.model), ('Myopic', self.model), ('Raw budget', None)):
            alpha = 0. if model is None else .1
            old, records, ledger = original_episode(self.trace, proposer, 'Energy', alpha,
                self.predictor, model, self.cfg)
            new, new_records, new_ledger = episode(self.trace, proposer, alpha, self.predictor,
                model, 1, self.cfg)
            self.assertEqual(old, new)
            self.assertEqual(records, new_records)
            for left, right in zip(ledger, new_ledger):
                for key in ('executed_bits', 'proposal_bits', 'reference_bits', 'accepted', 'reason'):
                    self.assertEqual(left[key], right[key])

    def test_minimum_four_command_invariance(self):
        for proposer, model in (('Value', self.model), ('Myopic', self.model), ('Raw budget', None)):
            _, records, ledger = episode(self.trace, proposer, .1, self.predictor, model, 4, self.cfg)
            self.assertTrue(all(r['command_bits'].bit_count() >= 4 for r in records))
            self.assertTrue(all(r['domain_size'] >= 13 for r in ledger))

    def test_mask_and_reference_validation(self):
        masks = np.array([np.arange(16) < n for n in range(1, 17)])
        ids = allowed(masks, 4)
        np.testing.assert_array_equal(ids, np.arange(3, 16))
        a, feasible = restricted_choice(np.arange(16.), np.zeros(16), 0., 3, ids)
        self.assertEqual(a, 3)
        self.assertEqual(feasible, 13)
        with self.assertRaises(ValueError):
            restricted_choice(np.arange(16.), np.zeros(16), 0., 0, ids)

    def test_raw_equivalence_and_restriction(self):
        env = Environment(self.cfg, self.trace)
        for _ in range(3):
            c = env.candidates()
            self.assertEqual(raw_action(c, 1), search_action(c))
            a = raw_action(c, 4)
            self.assertGreaterEqual(c.masks[a].sum(), 4)
            env.block(c.masks[a])


if __name__ == '__main__':
    unittest.main()
