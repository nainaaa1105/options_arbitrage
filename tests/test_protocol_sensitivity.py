from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from forward_validation.protocol import DEFAULT_ACCEPTANCE
from protocol_sensitivity import icc, one_trial


class SensitivityHelperTests(unittest.TestCase):
    def test_icc_is_one_for_group_constant_values_and_near_zero_for_independent_values(self):
        groups = np.repeat(np.arange(30), 6)
        constant = np.repeat(np.random.default_rng(1).normal(size=30), 6)
        rho, size = icc(constant, groups)
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertEqual(size, 6)
        noise = np.random.default_rng(2).normal(size=180)
        self.assertLess(icc(noise, groups)[0], .12)

    def test_a_clearly_better_candidate_has_a_positive_lower_bound_and_an_equal_one_does_not(self):
        rng = np.random.default_rng(3)
        reference = np.where(rng.random(360) < .1, -300., 0.) + rng.normal(0, 20, 360)
        days = np.repeat(np.arange(60), 6)
        better = np.maximum(reference, -80.)
        reduction, cost, lower = one_trial(better, reference, days, np.random.default_rng(4), 200)
        self.assertGreater(reduction, 0)
        self.assertGreater(lower, 0)
        _, _, equal_lower = one_trial(reference, reference, days, np.random.default_rng(4), 200)
        self.assertLessEqual(equal_lower, 0)

    def test_default_cost_ceiling_is_the_documented_value(self):
        self.assertEqual(DEFAULT_ACCEPTANCE["cost_efficiency"]["maximum_cost_per_unit_cvar_reduction"], .75)


if __name__ == "__main__":
    unittest.main()
