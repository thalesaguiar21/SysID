import unittest2 as unittest
import test_estimation as testim
import test_data as tdata


sestim = unittest.TestLoader().loadTestsFromModule(testim)
sdata = unittest.TestLoader().loadTestsFromModule(tdata)

suites = [sestim, sdata]

for suite in suites:
    unittest.TextTestRunner(verbosity=2).run(suite)
