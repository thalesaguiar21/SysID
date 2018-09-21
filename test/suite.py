import unittest2 as unittest
import test_estimation as testim


suite_estim = unittest.TestLoader().loadTestsFromModule(testim)
unittest.TextTestRunner(verbosity=2).run(suite_estim)
