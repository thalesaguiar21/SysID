import unittest
import tests.test_estimation as testim
import tests.test_data as tdata
import tests.test_identification as tid


sestim = unittest.TestLoader().loadTestsFromModule(testim)
sdata = unittest.TestLoader().loadTestsFromModule(tdata)
sid = unittest.TestLoader().loadTestsFromModule(tid)

suites = [sestim, sdata, sid]

for suite in suites:
    unittest.TextTestRunner(verbosity=2).run(suite)
