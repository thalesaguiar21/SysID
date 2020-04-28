import unittest

import numpy as np
from scipy.linalg import solve_toeplitz

from .context import sysid
from sysid.filtering import levdurb


class TestsLevdurb(unittest.TestCase):

    def test_solution_shape(self):
        margs = np.array([4, 3, 3, 1])
        coefs, error = levdurb.solve(margs)
        self.assertEqual(coefs.size, 4)

    def test_result_o4(self):
        linsys = np.array([1.0, 0.1, -0.8, -0.27])
        corr = np.array([0.3922, 0.0278, -0.0569, -0.0017])
        coefs, error = levdurb.solve(corr, corr.size - 1)
        toepl = np.array([[1, 0.1, -0.8, -0.27],
                          [0.1, 1, 0.1, -0.8],
                          [-0.8, 0.1, 1, 0.1],
                          [-0.27, -0.8, 0.1, 1]])
        pred = toepl @ coefs
        close = all(abs(p-r) < 0.15 for p, r in zip(pred, linsys))
        self.assertTrue(close, f"\n{pred}\n{linsys} -")

