"""Synthetic unit fixtures only, never manuscript experiment observations."""
import tempfile
import unittest
from pathlib import Path

import torch

from joint_power.risk_only.model import ReturnCritic, batches, frequency, load_checkpoint, losses


class CriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.linspace(-1, 1, 2*3*5).reshape(2, 3, 5)
        self.ref, self.remaining = torch.tensor([0, 2]), torch.tensor([20., 10.])
        self.outcome = torch.tensor([[[10., 0.], [11., 4.], [9., 2.]],
                                     [[6., 1.], [7., 3.], [8., 2.]]])

    def model(self, joint=False):
        return ReturnCritic(torch.zeros(5), torch.ones(5), .1, joint=joint)

    def test_same_trunk_and_independent_shuffle(self):
        torch.manual_seed(11)
        a = self.model()
        torch.manual_seed(11)
        b = self.model(True)
        for p, q in zip(a.trunk.parameters(), b.trunk.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        orders = []
        for extra in (1, 17):
            torch.rand(extra)
            gen = torch.Generator().manual_seed(22)
            orders.append(torch.cat([torch.cat(batches(33, gen)) for _ in range(3)]))
        self.assertTrue(torch.equal(*orders))

    def test_constant_initialization_capacity_and_anchor(self):
        for joint in (False, True):
            model = self.model(joint)
            count, _, energy = model(self.x, self.ref, self.remaining)
            torch.testing.assert_close(count, (.3*self.remaining[:, None]).expand_as(count))
            self.assertTrue(bool(torch.all(count <= 3*self.remaining[:, None])))
            if joint:
                torch.nn.init.normal_(model.head.weight)
                energy = model(self.x, self.ref, self.remaining)[2]
                torch.testing.assert_close(energy.gather(1, self.ref[:, None]), torch.zeros(2, 1))
            else:
                self.assertIsNone(energy)

    def test_risk_loss_identical_before_auxiliary_training(self):
        a, b = self.model(), self.model(True)
        la = losses(a, self.x, self.ref, self.remaining, self.outcome)
        lb = losses(b, self.x, self.ref, self.remaining, self.outcome)
        torch.testing.assert_close(la[0], lb[0], rtol=0, atol=0)
        self.assertEqual(float(la[1]), 0.)
        self.assertGreater(float(lb[1].detach()), 0.)
        sum(lb).backward()
        self.assertTrue(all(bool(torch.all(torch.isfinite(p.grad))) for p in b.parameters()))

    def test_capacity_weighted_frequency_and_invalid_labels(self):
        self.assertAlmostEqual(frequency(self.outcome, self.remaining), 12/270, places=7)
        invalid = self.outcome.clone()
        invalid[1, 0, 1] = 31
        with self.assertRaises(ValueError):
            frequency(invalid, self.remaining)

    def test_reload_exact(self):
        for joint in (False, True):
            model = self.model(joint)
            with tempfile.TemporaryDirectory(prefix='risk_only_test_') as tmp:
                path = Path(tmp)/'critic.pt'
                torch.save(dict(state=model.state_dict(), frequency=.1,
                    arm='joint' if joint else 'risk_only'), path)
                loaded, _ = load_checkpoint(path)
                for a, b in zip(model(self.x, self.ref, self.remaining),
                                loaded(self.x, self.ref, self.remaining)):
                    if a is not None:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_requires_positive_normalization_scale(self):
        with self.assertRaises(ValueError):
            ReturnCritic(torch.zeros(5), torch.zeros(5), .1)


if __name__ == '__main__':
    unittest.main()
