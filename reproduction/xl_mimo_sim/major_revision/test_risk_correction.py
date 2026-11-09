import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.risk_correction import RateRiskModel, predictions, select_action, true_hold_outcomes
from major_revision.risk_correction_study import causal_data, check_physical_hold


class RiskCorrectionTests(unittest.TestCase):
    def root(self):
        cfg = make_config('validation')
        run = FullDomainRollout(cfg, trajectory(cfg, 1160990001, (2., 1.)))
        for _ in range(32):
            run.step(baseline_command(run, 'cost', 1.08, 1) if run.t % 4 == 0 else None)
        run.prepare()
        return run

    def test_current_measured_channel_and_physical_hold_parity(self):
        run = self.root()
        data = causal_data(run)
        actual, _ = true_hold_outcomes(run.cfg, run.tr.channel[32:40],
            run.tr.requirements_mbps[32:40], data['available'])
        np.testing.assert_allclose(data['rates'][:, 0], actual[:, 0], rtol=1e-10, atol=1e-8)
        # All future slots deliberately retain the old forecasts.
        np.testing.assert_allclose(data['rates'][:, 1:], data['old_rates'][:, 1:], rtol=1e-12, atol=1e-9)
        re, ee = check_physical_hold(run.cfg, run.tr, run.t, run.hw, data, actual)
        self.assertLess(re, 1e-6)
        self.assertLess(ee, 1e-12)

    def test_old_search_actions_are_exactly_preserved(self):
        run = self.root()
        data = causal_data(run)
        for minimum in (1, 4):
            for margin in (.9, 1., 1.08, 1.24):
                action = select_action(data, run.req, 'old', margin, minimum)
                np.testing.assert_array_equal(data['masks'][action],
                    baseline_command(run, 'cost', margin, minimum))

    def test_no_hidden_future_or_mutation(self):
        run = self.root()
        before = run.clone()
        data = causal_data(run)
        changed = run.clone()
        channel, gain, req = run.tr.channel.copy(), run.tr.gain_db.copy(), run.tr.requirements_mbps.copy()
        channel[run.t+1:] *= 1e10
        gain[run.t+1:] = 500.
        req[run.t+1:] = 1e7
        changed.tr = replace(run.tr, channel=channel, gain_db=gain, requirements_mbps=req)
        other = causal_data(changed)
        for k in data:
            np.testing.assert_array_equal(data[k], other[k])
        np.testing.assert_array_equal(run.tracker.gain_db, before.tracker.gain_db)
        np.testing.assert_array_equal(run.hw.active, before.hw.active)
        self.assertEqual(run.metrics(), before.metrics())

    def test_fit_calibration_and_serialization(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(1160990002)
        x = rng.normal(size=(1000, 3))
        y = (rng.random(1000) < 1/(1+np.exp(-x[:, 0]))).astype(float)
        model = RateRiskModel.fit(x[:700], y[:700])
        mean = model.mean.copy()
        model.calibrate(x[700:], y[700:])
        self.assertGreater(model.temperature, 0)
        np.testing.assert_array_equal(model.mean, mean)
        p = model.probability(x)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'risk.npz'
            model.save(path)
            np.testing.assert_array_equal(p, RateRiskModel.load(path).probability(x))

    def test_known_current_risk_override(self):
        run = self.root()
        data = causal_data(run)
        d = data['features'].shape[-1]
        model = RateRiskModel(np.zeros(d), np.ones(d), np.zeros(d), 0.)
        p = model.predict(data, run.req)
        np.testing.assert_array_equal(p[:, 0], data['rates'][:, 0] < run.req)
        self.assertTrue(np.all(p[data['available'].sum(2) == 0] == 1))


if __name__ == '__main__':
    unittest.main()
