import tempfile
import unittest
from pathlib import Path

import numpy as np

from future_vr.config import make_config
from major_revision.information_probe import causal_snapshot, probe_choices
from major_revision.probe_predictability import (
    FEATURE_NAMES, KINDS, RIDGE, OutcomePredictor, features, choose, kernel,
)
from major_revision.probe_predictability_study import SOURCE, snapshot


class ProbePredictabilityTests(unittest.TestCase):
    def sample(self):
        return snapshot(SOURCE/'shard0/root_s1_i0_t32.npz')

    def test_features_have_no_hidden_input_and_do_not_mutate(self):
        cfg, snap = make_config('validation'), self.sample()
        before = {k: v.copy() for k, v in snap.items()}
        q, x = features(cfg, snap, 32)
        snap['future_truth'] = np.full((140, 16, 3), 1e100)
        snap['scenario_id'] = np.array(999999)
        qq, xx = features(cfg, snap, 32)
        np.testing.assert_array_equal(q, qq)
        np.testing.assert_array_equal(x, xx)
        self.assertEqual(x.shape[1], len(FEATURE_NAMES))
        for key, value in before.items():
            np.testing.assert_array_equal(snap[key], value)

    def test_training_scaler_and_targets(self):
        x = np.arange(24.).reshape(8, 3)
        y = np.column_stack((np.arange(8.), -np.arange(8.)))
        y_copy = y.copy()
        for kind in KINDS:
            model = OutcomePredictor.fit(x, y, kind)
            np.testing.assert_array_equal(model.mean, x.mean(0))
            mean, scale = model.mean.copy(), model.scale.copy()
            model.predict(x*1000)
            np.testing.assert_array_equal(model.mean, mean)
            np.testing.assert_array_equal(model.scale, scale)
            np.testing.assert_array_equal(y, y_copy)

    def test_closed_form_equations_and_serialization(self):
        rng = np.random.default_rng(20260921)
        x, y = rng.normal(size=(19, 4)), rng.normal(size=(19, 2))
        for kind in KINDS:
            model = OutcomePredictor.fit(x, y, kind)
            z = (x-model.mean)/model.scale
            target = y-y.mean(0)
            if kind == 'linear':
                residual = (z.T@z+RIDGE*np.eye(4))@model.weights-z.T@target
            else:
                residual = (kernel(z, z)+RIDGE*np.eye(19))@model.weights-target
            self.assertLess(np.max(np.abs(residual)), 1e-10)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.npz'
                model.save(path)
                np.testing.assert_array_equal(model.predict(x), OutcomePredictor.load(path).predict(x))

    def test_constant_target_and_empty_candidates(self):
        x = np.ones((5, 3))
        y = np.tile([.33, 0.], (5, 1))
        for kind in KINDS:
            model = OutcomePredictor.fit(x, y, kind)
            np.testing.assert_allclose(model.predict(x), y, atol=1e-12)
            self.assertEqual(model.predict(np.empty((0, 3))).shape, (0, 2))
        self.assertEqual(choose([], np.empty((0, 2))), -1)

    def test_skip_risk_filter_and_tie(self):
        q = np.array([7, 4, 9])
        self.assertEqual(choose(q, np.array([[.1, -1.], [-1., .1], [0., 0.]])), -1)
        self.assertEqual(choose(q, np.array([[-1., 0.], [-1., -.1], [-2., .1]])), 4)

    def test_snapshot_roundtrip_of_causal_features(self):
        from major_revision.action_value import baseline_command
        from major_revision.full_domain import FullDomainRollout
        from major_revision.pilot import trajectory
        cfg = make_config('validation')
        run = FullDomainRollout(cfg, trajectory(cfg, 1140010000, (2., 1.)))
        while run.t < 32:
            ref = baseline_command(run, 'cost', 1.08, 1) if run.t % 4 == 0 else None
            run.step(ref)
        ref = baseline_command(run, 'cost', 1.08, 1)
        _, choices = probe_choices(run, ref)
        live = causal_snapshot(run, ref, choices)
        q, x = features(cfg, live, run.t)
        qq, xx = features(cfg, self.sample(), run.t)
        np.testing.assert_array_equal(q, qq)
        np.testing.assert_array_equal(x, xx)


if __name__ == '__main__':
    unittest.main()
