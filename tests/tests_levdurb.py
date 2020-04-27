import unittest

import numpy as np
from scipy.linalg import solve_toeplitz

from .context import sysid
from sysid.filtering import levdurb


class TestsLevdurb(unittest.TestCase):

    def setUp(self):
        pass

    def test_solution_shape(self):
        margs = np.ones(4)
        coefs, error = levdurb.solve(margs)
        self.assertEqual(coefs.size, 4)

    def test_result(self):
        linsys = np.array([1.0, 0.1, -0.8, -0.27])
        corr = np.array([0.4384, 0.0917, -0.0659, -0.0172])
        coefs, error = levdurb.solve(corr)
        toepl = np.array([[1, 0.1, -0.8, -0.27],
                          [0.1, 1, 0.1, -0.8],
                          [-0.8, 0.1, 1, 0.1],
                          [-0.27, -0.8, 0.1, 1]])
        pred = toepl @ coefs
        breakpoint()
        close = all(abs(p-r) < 0.001 for p, r in zip(pred, linsys))
        self.assertTrue(close, f"\n{pred}\n{linsys} -")
