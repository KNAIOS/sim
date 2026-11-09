import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.action_value import observation,full_slot_energy
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.structural_value_audit import known_costs,serialized_energy


class StructuralValueTests(unittest.TestCase):
    def test_known_cost_uses_only_causal_csi_and_matches_executed_block(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        run=FullDomainRollout(cfg,trajectory(cfg,1040902001,(2.,1.)))
        _,score=observation(run)
        run.step(select_top(score,4))
        for _ in range(3):
            run.step()
        before,score=observation(run)
        energy,outage=known_costs(run,score)
        for action in range(16):
            trial=run.clone()
            initial=trial.energy
            trial.step(select_top(score,action+1))
            self.assertAlmostEqual(outage[action],trial.outages[-1],places=12)
            for _ in range(3):
                trial.step()
            self.assertAlmostEqual(energy[action],(trial.energy-initial)/(4*full_slot_energy(cfg)),places=12)
        np.testing.assert_array_equal(observation(run)[0],before)
        extended=np.concatenate((before,[1.08,1/16]))[None].astype(np.float32)
        np.testing.assert_allclose(serialized_energy(cfg,extended,np.array([3]))[0],energy-energy[3],atol=1e-7)


if __name__=='__main__':
    unittest.main()
