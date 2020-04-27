import unittest

import numpy as np

from .context import sysid
from sysid.filtering import levdurb


class TestsLevdurb(unittest.TestCase):

    def setUp(self):
        pass

    def test_levdurb(self):
        margs = np.ones(4)
        icog = levdurb.solve(margs)

