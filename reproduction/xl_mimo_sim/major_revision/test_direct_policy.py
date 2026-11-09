import unittest
from dataclasses import replace
import numpy as np

from future_vr.config import make_config
from major_revision.absolute_return_screen import search_command
from major_revision.action_value import full_slot_energy
from major_revision.direct_policy import decision, episode, margin_value, ars_update
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory


class DirectPolicyTests(unittest.TestCase):
    def test_neutral_exactly_matches_search_and_preserves_real_requirements(self):
        cfg = replace(make_config('validation'), episode_steps=24)
        tr = trajectory(cfg, 1120900001, (2., 1.))
        original = tr.requirements_mbps.copy()
        for margin in (.85, 1.04, 1.35):
            run, other = FullDomainRollout(cfg, tr), FullDomainRollout(cfg, tr)
            while run.t < cfg.episode_steps:
                if run.t % 4:
                    run.step()
                    other.step()
                    continue
                command, _, actual_margin, _ = decision(run, np.zeros(9), margin)
                expected = search_command(other, 'cost', margin, 1)
                np.testing.assert_array_equal(command, expected)
                self.assertEqual(actual_margin, margin)
                run.step(command)
                other.step(expected)
            self.assertEqual(run.metrics(), other.metrics())
        np.testing.assert_array_equal(original, tr.requirements_mbps)

    def test_no_future_or_unobserved_current_information(self):
        cfg = replace(make_config('validation'), episode_steps=24)
        tr = trajectory(cfg, 1120900002, (3., 1.15))
        run = FullDomainRollout(cfg, tr)
        run.step(np.arange(cfg.subarrays) < 4)
        for _ in range(3):
            run.step()
        clone = run.clone()
        # Clone the exogenous object, because the production clone shares it.
        import copy
        clone.tr = copy.deepcopy(run.tr)
        clone.tr.channel[run.t+1:] = np.nan
        clone.tr.gain_db[run.t+1:] = np.nan
        clone.tr.requirements_mbps[run.t+1:] = np.nan
        sleeping = ~clone.hw.active
        antennas = np.repeat(sleeping, cfg.antennas_per_subarray)
        # Use finite extreme hidden values: np.where intermediates may otherwise
        # evaluate unused NaNs although actual measurements are correctly masked.
        clone.tr.channel[run.t, :, antennas] = 1000.
        clone.tr.gain_db[run.t, sleeping] = 1000.
        a = decision(run, np.arange(9)*.01)
        b = decision(clone, np.arange(9)*.01)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2:], b[2:])

    def test_bounds_bias_equivalence_and_reject_nonfinite(self):
        phi = np.array([1., .2, -.3, .4, .5, .1, .2, .7, 0.])
        self.assertEqual(margin_value(np.ones(9)*1000, phi), 1.35)
        self.assertEqual(margin_value(np.ones(9)*-1000, phi), .85)
        self.assertEqual(margin_value(np.array([.1]), phi),
                         margin_value(np.array([.1]+[0.]*8), phi))
        with self.assertRaises(ValueError):
            margin_value(np.array([np.nan]), phi)

    def test_update_ascends_reward_and_flat_return_does_not_move(self):
        theta = np.array([0.])
        delta = np.array([[1.], [-1.], [.5], [-.5]])
        reward = np.column_stack((-(.08*delta[:,0]-1.)**2,
                                  -(-.08*delta[:,0]-1.)**2))
        updated, order, sigma = ars_update(theta, delta, reward)
        self.assertGreater(updated[0], 0.)
        self.assertLess((updated[0]-1.)**2, 1.)
        self.assertEqual(len(order), 2)
        self.assertGreater(sigma, 0.)
        unchanged, _, _ = ars_update(theta, delta, np.ones((4,2)))
        np.testing.assert_array_equal(unchanged, theta)

    def test_episode_energy_accounting_and_replay(self):
        cfg = replace(make_config('validation'), episode_steps=20)
        tr = trajectory(cfg, 1120900003, (.5, .85))
        result, commands = episode(cfg, tr, np.arange(9)*.01, record=True)
        replay = FullDomainRollout(cfg, tr)
        for row in commands:
            replay.step(np.asarray(row['command'], bool))
            for _ in range(3):
                replay.step()
        for key, value in replay.metrics().items():
            self.assertEqual(result[key], value)
        self.assertAlmostEqual(result['energy_j'], result['operating_j']+
                               result['switching_j']+result['pilot_j'])
        self.assertGreater(full_slot_energy(cfg), 0.)


if __name__ == '__main__':
    unittest.main()
