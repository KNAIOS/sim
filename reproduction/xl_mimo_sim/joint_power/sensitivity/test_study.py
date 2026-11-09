"""Protocol and baseline equivalence; synthetic fixtures never enter results."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from future_vr.config import make_config
from joint_power.anchor_confirmation.domain_prs import command_at, episode as expanded_episode
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import load_predictor
from joint_power.matched_four.control import episode as learned_episode
from joint_power.model import _prediction
from joint_power.model import Environment
from joint_power.risk_budget.control import episode as budget_episode
from joint_power.rollout.collect import base_command
from joint_power.sensitivity.study import (
    LEARNED, NUMBER, SETTINGS, comparisons, reference_episode, specifications,
)
from joint_power.study import make_trace


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_config('validation')
        # Test-only seed; distinct from the frozen experiment stream.
        cls.trace = make_trace(cls.cfg, 5799220010, (2., 1.))

    def test_frozen_design(self):
        self.assertEqual(len(SETTINGS), 5)
        self.assertEqual(len(set(SETTINGS)), 5)
        self.assertEqual(SETTINGS[0], (self.cfg.uncertainty_penalty, self.cfg.visibility_discount))
        self.assertEqual(len(specifications()), 8)
        self.assertEqual((len(SETTINGS)*len(specifications())+1)*3*NUMBER, 984)
        self.assertEqual({m for m, seed in specifications() if seed}, set(LEARNED))

    def test_default_original_prs_exact_episode(self):
        env = Environment(self.cfg, self.trace)
        while env.t < self.cfg.episode_steps:
            env.block(base_command(env))
        metrics, records = reference_episode(self.cfg, self.trace, 'PRS4')
        self.assertEqual(records, env.records)
        self.assertEqual(metrics, env.metrics())

    def test_default_budget_prs_exact_episode(self):
        old, records, _ = budget_episode(self.trace, 'PRS budget', 1., None, cfg=self.cfg)
        new, new_records = reference_episode(self.cfg, self.trace, 'budgetPRS4')
        self.assertEqual(old, new)
        self.assertEqual(records, new_records)

    def test_expanded_references_exact_episode(self):
        for adaptive, name in ((False, 'PRS1'), (True, 'budgetPRS1')):
            old, records = expanded_episode(self.trace, adaptive)
            new, new_records = reference_episode(self.cfg, self.trace, name)
            self.assertEqual(old, new)
            self.assertEqual(records, new_records)

    def test_perturbed_config_reaches_each_reference(self):
        cfg = replace(self.cfg, episode_steps=8, uncertainty_penalty=1.3, visibility_discount=.8)
        for name in ('PRS4', 'budgetPRS4', 'PRS1', 'budgetPRS1'):
            with patch('joint_power.sensitivity.study.command_at', wraps=command_at) as observed:
                reference_episode(cfg, self.trace, name)
            self.assertEqual(observed.call_count, 2)
            for call in observed.call_args_list:
                actual = call.args[0]
                self.assertEqual(actual.uncertainty_penalty, 1.3)
                self.assertEqual(actual.visibility_discount, .8)
                self.assertEqual(call.kwargs['minimum'], int(name[-1]))

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            reference_episode(self.cfg, self.trace, 'unknown')

    def test_perturbed_config_reaches_learned_forecast(self):
        cfg = replace(self.cfg, episode_steps=8, uncertainty_penalty=1.3, visibility_discount=.8)
        with patch('joint_power.model._prediction', wraps=_prediction) as observed:
            _, records, _ = learned_episode(self.trace, 'Value', .1, load_predictor(),
                models()[0]['Visitation-added', 11], 4, cfg)
        self.assertEqual(len(records), 8)
        self.assertGreater(observed.call_count, 0)
        for call in observed.call_args_list:
            self.assertEqual(call.args[0], cfg)

    def test_paired_bootstrap_preserves_constant_fixture(self):
        rows = []
        for scenario in range(3):
            for setting in range(len(SETTINGS)):
                for method, seed in specifications():
                    for i in range(NUMBER):
                        learned = method in LEARNED
                        rows.append(dict(scenario=scenario, setting=setting, method=method,
                            seed=seed, trajectory=i, energy_j=10.*i+setting+int(learned),
                            outage=.01*setting+.02*int(learned)))
        paired = comparisons(rows)
        self.assertEqual(len(paired), 108)
        for row in paired:
            expected = (1. if row['metric'] == 'energy_j' else .02)
            if row['comparison'] == 'default parameter':
                expected = row['setting']*(1. if row['metric'] == 'energy_j' else .01)
            for key in ('difference', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(row[key], expected, places=12)


if __name__ == '__main__':
    unittest.main()
