import copy
import unittest
import numpy as np
from types import SimpleNamespace
from major_revision.constraint_training import preserved_exploration_rng, selection_key


class ConstraintTrainingTests(unittest.TestCase):
    def test_feasible_model_beats_lower_energy_infeasible_model(self):
        safe = [dict(scenario=j, outage=.04, energy_j=210.) for j in range(3)]
        cheap = [dict(scenario=j, outage=.0 if j < 2 else .08, energy_j=180.) for j in range(3)]
        self.assertLess(selection_key(safe)[:2], selection_key(cheap)[:2])

    def test_worst_scenario_not_pooled_outage(self):
        data = [dict(scenario=j, outage=.12 if j == 2 else 0., energy_j=200.) for j in range(3)]
        violation, _, worst = selection_key(data)
        self.assertAlmostEqual(worst, .12)
        self.assertAlmostEqual(violation, .07)

    def test_validation_does_not_advance_training_rng(self):
        model = SimpleNamespace(rng=np.random.default_rng(31))
        before = copy.deepcopy(model.rng.bit_generator.state)
        with preserved_exploration_rng(model):
            model.rng.random(100)
        self.assertEqual(model.rng.bit_generator.state, before)

    def test_validation_exception_restores_rng(self):
        model = SimpleNamespace(rng=np.random.default_rng(31))
        before = copy.deepcopy(model.rng.bit_generator.state)
        with self.assertRaises(RuntimeError):
            with preserved_exploration_rng(model):
                model.rng.random(10)
                raise RuntimeError('test')
        self.assertEqual(model.rng.bit_generator.state, before)


if __name__ == '__main__':
    unittest.main()
