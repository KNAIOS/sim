import unittest
from dataclasses import replace
import numpy as np

from future_vr.config import make_config
from major_revision.action_value import baseline_command
from major_revision.full_domain import FullDomainRollout
from major_revision.information_probe import (ProbeRollout, probe_choices, probe_command,
                                              branch, independent_reference)
from major_revision.pilot import trajectory


class InformationProbeTests(unittest.TestCase):
    def root(self):
        cfg = replace(make_config('validation'), episode_steps=36)
        run = FullDomainRollout(cfg, trajectory(cfg, 1140900001, (2., 1.)))
        # Deterministic legal history creates sleeping candidates without a future choice.
        mask = np.zeros(cfg.subarrays, bool)
        mask[:5] = True
        for t in range(8):
            run.step(mask if t % 4 == 0 else None)
        run.prepare()
        return run, mask

    def test_normal_adapter_exact_parity(self):
        cfg = replace(make_config('validation'), episode_steps=24)
        tr = trajectory(cfg, 1140900002, (2., 1.))
        old, new = FullDomainRollout(cfg, tr), ProbeRollout(cfg, tr)
        for t in range(cfg.episode_steps):
            command = None
            if t % 4 == 0:
                command = baseline_command(old, 'cost', 1.08, 1)
                np.testing.assert_array_equal(command, baseline_command(new, 'cost', 1.08, 1))
            old.step(command)
            new.step(command)
            np.testing.assert_array_equal(old.tracker.channel_estimate, new.tracker.channel_estimate)
            np.testing.assert_array_equal(old.tracker.gain_db, new.tracker.gain_db)
            np.testing.assert_array_equal(old.age, new.age)
        self.assertEqual(old.metrics(), new.metrics())

    def test_choices_use_no_hidden_future_or_current_sleeping_csi(self):
        root, ref = self.root()
        eligible, choices = probe_choices(root, ref)
        changed = root.clone()
        gains, channels = root.tr.gain_db.copy(), root.tr.channel.copy()
        gains[root.t+1:] = 500.
        channels[root.t+1:] *= 1e8
        gains[root.t, eligible] = -500.
        changed.tr = replace(root.tr, gain_db=gains, channel=channels)
        candidates, other = probe_choices(changed, ref)
        np.testing.assert_array_equal(eligible, candidates)
        self.assertEqual(choices, other)
        self.assertTrue(np.all(~root.hw.active[eligible]))

    def test_no_instant_probe_and_same_physical_cost_for_withholding(self):
        root, ref = self.root()
        q = 8
        first = probe_command(root, ref, q)
        normal = ProbeRollout.from_root(root)
        blind = ProbeRollout.from_root(root, q, root.t+4, root.t+8)
        initial_q = root.tracker.channel_estimate[:, q*8:(q+1)*8].copy()
        for h in range(12):
            for run in (normal, blind):
                run.prepare()
            if h < 4:
                self.assertFalse(normal.observed[q])
                np.testing.assert_array_equal(normal.tracker.channel_estimate[:, q*8:(q+1)*8], initial_q)
            if h == 4:
                self.assertTrue(normal.observed[q])
                self.assertEqual(normal.age[q], 0.)
                self.assertGreater(blind.age[q], 0.)
                np.testing.assert_array_equal(blind.tracker.channel_estimate[:, q*8:(q+1)*8], initial_q)
            if h == 8:
                self.assertEqual(blind.age[q], 0.)
                np.testing.assert_array_equal(normal.tracker.channel_estimate, blind.tracker.channel_estimate)
            for run in (normal, blind):
                run.step(first if run.t % 4 == 0 else None)
            # Given identical commands, information suppression cannot change the PHY or costs.
            self.assertEqual(normal.metrics(), blind.metrics())

    def test_probe_first_block_cost_and_reference_parity(self):
        root, ref = self.root()
        base = branch(root, ref, -1, 1.08, 1)
        original = independent_reference(root, ref, 1.08, 1)
        for key in original:
            if key != 'method':
                self.assertEqual(base[key], original[key])
        for withheld in (False, True):
            result = branch(root, ref, 8, 1.08, 1, withheld)
            self.assertAlmostEqual(result['first_block_energy_j']-base['first_block_energy_j'], .33, places=11)
            self.assertEqual(result['first_block_outage'], base['first_block_outage'])

    def test_branch_does_not_mutate_root_or_external_data(self):
        root, ref = self.root()
        before = root.clone()
        channel = root.tr.channel.copy()
        branch(root, ref, 8, 1.08, 1, True)
        self.assertEqual(root.metrics(), before.metrics())
        np.testing.assert_array_equal(root.tracker.channel_estimate, before.tracker.channel_estimate)
        np.testing.assert_array_equal(root.hw.active, before.hw.active)
        np.testing.assert_array_equal(root.tr.channel, channel)

    def test_no_candidate_and_invalid_probe(self):
        root, _ = self.root()
        full = np.ones(root.cfg.subarrays, bool)
        eligible, choices = probe_choices(root, full)
        self.assertEqual(len(eligible), 0)
        self.assertEqual(choices, dict(score=-1, age=-1))
        with self.assertRaises(ValueError):
            probe_command(root, full, 0)
        np.testing.assert_array_equal(probe_command(root, full, -1), full)


if __name__ == '__main__':
    unittest.main()
