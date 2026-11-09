import unittest
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.aggregation.collect import visitor_command
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.rollout.collect import base_command, state_labels
from joint_power.rollout.control import episode, load_models
from joint_power.study import make_trace


class AggregationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.predictor = load_predictor()

    def test_visitor_reproduces_previous_implementation(self):
        cfg = make_config('validation')
        trace = make_trace(cfg, 4099220000, (2., 1.))
        models, _ = load_models()
        for seed, price in ((11, 4.), (22, 2.)):
            model = models['MC absolute', seed]
            env = Environment(cfg, trace)
            while env.t < cfg.episode_steps:
                env.block(visitor_command(env, self.predictor, model, price))
            expected, records = episode(trace, 'MC absolute', price, self.predictor, model)
            self.assertEqual(env.records, records)
            self.assertEqual(env.metrics(), expected)

    def test_below_four_root_reference_is_causal_prs_suffix(self):
        cfg = replace(make_config('validation'), episode_steps=20)
        trace = make_trace(cfg, 4099220001, (2., 1.))
        env = Environment(cfg, trace)
        env.block(env.candidates().masks[1])
        self.assertEqual(int(env.hw.active.sum()), 2)
        labels = state_labels(env, self.predictor)
        start = len(env.records)
        while env.t < cfg.episode_steps:
            env.block(base_command(env))
        suffix = env.records[start:]
        expected = [sum(r['energy_j'] for r in suffix), sum(round(3*r['outage']) for r in suffix)]
        np.testing.assert_allclose(labels['y'][labels['reference'], -1], expected, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
