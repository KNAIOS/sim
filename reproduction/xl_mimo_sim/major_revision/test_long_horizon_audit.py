import unittest
import numpy as np
from major_revision.long_horizon_audit import horizon_returns, error_parts


class LongHorizonAuditTests(unittest.TestCase):
    def test_returns_use_same_discount_origin_and_units(self):
        blocks = np.ones((2, 3, 16, 2))
        result = horizon_returns(blocks, .98)
        self.assertEqual(result.shape, (2, 3, 4, 2))
        for h, count in enumerate((1, 4, 8, 16)):
            np.testing.assert_allclose(result[:, :, h], sum(.98**n for n in range(count)))
        for invalid in ((0,), (5,), (68,)):
            with self.assertRaises(ValueError):
                horizon_returns(blocks, .98, invalid)

    def test_prediction_and_truncation_mse_need_cross_term(self):
        parts = error_parts([3., 5.], [1., 2.], [3., 5.])
        self.assertEqual(parts['error_to_C64_mse'], 0.)
        self.assertEqual(parts['prediction_error_mse'], 6.5)
        self.assertEqual(parts['omitted_tail_mse'], 6.5)
        self.assertEqual(parts['cross_term'], -13.)
        self.assertEqual(parts['identity_error'], 0.)
        np.testing.assert_allclose(error_parts([1., 1.], [0., 0.], [-1., -2.])['cross_term'], 3.)


if __name__ == '__main__':
    unittest.main()
