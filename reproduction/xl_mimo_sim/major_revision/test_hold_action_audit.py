import unittest
from dataclasses import replace
import numpy as np
import torch

from future_vr.config import make_config
from major_revision.full_domain import FullDomainRollout
from major_revision.hold_action_audit import augmented_data, choose, physical_check, branch
from major_revision.pilot import trajectory
from major_revision.risk_correction import select_action
from major_revision.risk_correction_study import causal_data
from major_revision.risk_rl_study import setup


class HoldActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.settings, cls.model = setup()

    def root(self):
        cfg = make_config('validation')
        run = FullDomainRollout(cfg, trajectory(cfg, 1220990001, (2., 1.)))
        while run.t < 24:
            if run.t % 4:
                run.step()
            else:
                data = causal_data(run)
                action = select_action(data, run.req, 'calibrated', *self.settings['calibrated', 1], self.model)
                run.step(data['masks'][action])
        run.prepare()
        return run

    def test_prefix_rows_and_choices_unchanged_additional_physical_parity(self):
        run = self.root()
        old, new = causal_data(run), augmented_data(run)
        for k in old:
            np.testing.assert_array_equal(old[k], new[k][:-1])
        np.testing.assert_array_equal(new['masks'][-1], run.hw.command)
        for mode in ('old', 'anchored', 'calibrated'):
            for minimum in (1, 4):
                parameter = self.settings[mode, 1][0]
                self.assertEqual(select_action(old, run.req, mode, parameter, minimum, self.model),
                    choose(new, run.req, mode, parameter, minimum, self.model, False))
        error, current = physical_check(run, new)
        self.assertLess(error, 1e-10)
        self.assertLess(current, 1e-6)

    def test_no_hidden_future_or_input_mutation(self):
        run = self.root()
        old = run.clone()
        result = augmented_data(run)
        changed = run.clone()
        h, g, r = run.tr.channel.copy(), run.tr.gain_db.copy(), run.tr.requirements_mbps.copy()
        h[run.t+1:] *= 1e8
        g[run.t+1:] = 1000.
        r[run.t+1:] = 1e8
        changed.tr = replace(run.tr, channel=h, gain_db=g, requirements_mbps=r)
        for k, v in augmented_data(changed).items():
            np.testing.assert_array_equal(v, result[k])
        np.testing.assert_array_equal(run.hw.command, old.hw.command)
        np.testing.assert_array_equal(run.hw.countdown, old.hw.countdown)
        np.testing.assert_array_equal(run.tracker.gain_db, old.tracker.gain_db)
        self.assertEqual(run.metrics(), old.metrics())

    def test_cardinality_filter_and_tie_not_action_index(self):
        masks = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]], bool)
        data = dict(masks=masks, energy=np.array([[4.], [3.], [2.], [1.]]),
                    old_rates=np.full((4, 1, 1), 2.))
        self.assertEqual(choose(data, np.array([1.]), 'old', 1., 1, None, True), 3)
        self.assertEqual(choose(data, np.array([1.]), 'old', 1., 2, None, True), 2)
        data['energy'][-1] = 2.
        self.assertEqual(choose(data, np.array([1.]), 'old', 1., 1, None, True), 2)

    def test_branch_identity_and_root_not_mutated(self):
        root = self.root()
        data = augmented_data(root)
        args = ('calibrated', *self.settings['calibrated', 1], self.model)
        a = choose(data, root.req, *args, False)
        before = root.metrics()
        result = branch(root, data['masks'][a], *args)
        reference = root.clone()
        while reference.t < reference.cfg.episode_steps:
            if reference.t % 4:
                reference.step()
            else:
                v = causal_data(reference)
                index = select_action(v, reference.req, *args)
                reference.step(v['masks'][index])
        self.assertEqual(result, reference.metrics())
        self.assertEqual(root.metrics(), before)


if __name__ == '__main__':
    unittest.main()
