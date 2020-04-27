import unittest

import numpy as np

from .context import sysid
from sysid.filtering import levdurb


class TestsCholesky(unittest.TestCase):

    def setUp(self):
        pass

    def test_cholesky(self):
        margs = np.ones(4)
        icog = levdurb.solve(margs)

    def test_diagonal(self):
        margs = np.ones((4, 4))
        diag = levdurb.diag(margs)
        diag_is_one = all(diag[i, i] == 1 for i in range(4))
        out_is_zero = True
        for i in range(4):
            for j in range(4):
                if i != j:
                    out_is_zero &= diag[i, j] == 0
        self.assertTrue(diag_is_one and out_is_zero)

