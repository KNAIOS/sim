import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.anchor_confirmation.domain_prs import command_at
from joint_power.model import Environment
from joint_power.study import make_trace
from major_revision.budget_search import search


class DomainTests(unittest.TestCase):
    def test_original_floor_reproduces_original_search(self):
        cfg = replace(make_config('validation'), episode_steps=20)
        env = Environment(cfg, make_trace(cfg, 4299220007, (2., 1.)))
        for _ in range(4):
            env.prepare()
            old = search(cfg, env.tracker, env.hw, env.req)
            new = command_at(cfg, env.tracker, env.hw, env.req, minimum=4)
            np.testing.assert_array_equal(new, old)
            env.block(old)


if __name__ == '__main__':
    unittest.main()
