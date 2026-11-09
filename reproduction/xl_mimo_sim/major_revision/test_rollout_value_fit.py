import unittest
import numpy as np
import torch
from future_vr.config import make_config
from major_revision.rollout_value_fit import DifferenceNetwork,select_action,metrics


class RolloutValueFitTests(unittest.TestCase):
    def test_anchor_fallback_and_learned_update(self):
        torch.manual_seed(11)
        cfg=make_config('validation')
        model=DifferenceNetwork(cfg)
        x=torch.randn(4,1436)
        ref=torch.tensor([1,3,7,15])
        before=model(x,ref)
        self.assertEqual(before.shape,(4,16,2))
        self.assertEqual(sum(p.numel() for p in model.parameters()),32066)
        np.testing.assert_array_equal(before.detach(),0.)
        for i in range(4):
            self.assertEqual(select_action(before[i].detach(),int(ref[i]),2.),ref[i])
        optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4)
        truth=torch.randn_like(before)
        ((before-truth)**2).mean().backward()
        optimizer.step()
        after=model(x,ref).detach().numpy()
        np.testing.assert_array_equal(after[np.arange(4),ref],0.)
        self.assertGreater(np.abs(after).sum(),0.)
        result=metrics(after,truth.numpy(),ref.numpy())
        self.assertTrue(all(np.isfinite(list(result.values()))))

    def test_selection_uses_cost_not_larger_reward_and_rejects_nan(self):
        values=np.array([[.1,-.1],[0.,0.],[-.03,.02]])
        self.assertEqual(select_action(values,1,2.),0)
        self.assertEqual(select_action(values,1,1.),2)
        values[0,0]=np.nan
        with self.assertRaises(ValueError):
            select_action(values,1,2.)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
