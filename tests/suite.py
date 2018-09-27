from tests import test_data as tdata
from tests import test_estimation as testim
from tests import test_identification as tid
import unittest


def run():
    sestim = unittest.TestLoader().loadTestsFromModule(testim)
    sdata = unittest.TestLoader().loadTestsFromModule(tdata)
    sid = unittest.TestLoader().loadTestsFromModule(tid)

    suites = [sestim, sdata, sid]

    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)
