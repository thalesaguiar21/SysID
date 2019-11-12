import unittest
import numpy as np

from .context import sysid
from sysid.filtering import lse



def assertSequenceAlmostEqual(testcase, seq1, seq2, tol):
    for x, y in zip(seq1, seq2):
        testcase.assertAlmostEqual(x, y, tol)


class TestEstimation(unittest.TestCase):

    def setUp(self):
        self.coef = np.array('1 2; 1 2')
        self.rs = np.array('1; 2')
        est.lse.initial_confidence = 1000
        est.lse.forget_rate = 1.0

    def test_none_rs(self):
        self.rs = None
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('System estimated with None rs matrix')
        except ValueError:
            pass

    def test_empty_rs(self):
        self.rs = np.array(' ')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('System estimated with empty rs matrix')
        except ValueError:
            pass

    def test_none_coef(self):
        self.coef = None
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('System estimated with None coefficient matrix')
        except ValueError:
            pass

    def test_empty_coef(self):
        self.coef = np.array('')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('System estimated with empty coefficient matrix')
        except ValueError:
            pass

    def test_indetermined_sys(self):
        self.coef = np.array('1 2 3; 1 2 3')
        self.rs = np.array('1; 2')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('Tried to solve an indetermined system!')
        except ValueError:
            pass

    def test_rs_small(self):
        self.coef = np.array('1 2 3; 1 2 3')
        self.rs = np.array('1')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('Tried to solve with insuficient result values!')
        except ValueError:
            pass

    def test_rs_larger(self):
        self.coef = np.array('1 2 3; 1 2 3')
        self.rs = np.array('1; 2; 3')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('Tried to solve with too many result values!')
        except ValueError:
            pass

    def test_column_rs(self):
        self.coef = np.array('1 2 3; 1 2 3')
        self.rs = np.array('1; 2')
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail('Tried to solve with column result vector!')
        except ValueError:
            pass

    def try_solve_recursive(self, msg):
        try:
            params = est.lse.recursive(self.coef, self.rs)
            self.fail(msg)
        except ValueError:
            pass

    def test_ffactor_upbounds(self):
        self.ffac = 1.5
        p_15 = est.lse.recursive(self.coef, self.rs)
        self.ffac = 1.0
        p_10 = est.lse.recursive(self.coef, self.rs)
        p_15 = [elm[0] for elm in p_15]
        p_10 = [elm[0] for elm in p_10]
        self.assertSequenceEqual(p_15, p_10)

    def test_ffactor_lbounds_out(self):
        self.ffac = 1e-4
        p_1 = est.lse.recursive(self.coef, self.rs)
        self.ffac = -1.0
        p_2 = est.lse.recursive(self.coef, self.rs)
        p_1 = [elm[0] for elm in p_1]
        p_2 = [elm[0] for elm in p_2]
        self.assertSequenceEqual(p_1, p_2)

    def test_ffactor_lbounds_cout(self):
        self.ffac = 1e-4
        p_1 = est.lse.recursive(self.coef, self.rs)
        self.ffac = 1e-5
        p_2 = est.lse.recursive(self.coef, self.rs)
        p_1 = [elm[0] for elm in p_1]
        p_2 = [elm[0] for elm in p_2]
        self.assertSequenceEqual(p_1, p_2)

    def test_determined_sys(self):
        self.coef = np.array('2 3 2; 1 3 2; 1 2 2')
        self.rs = np.array('12; 13; 11')
        params = est.lse.recursive(self.coef, self.rs)
        result = np.dot(self.coef, params)
        result = [e[0, 0] for e in result]
        expected = [12.002992517583628, 12.998997523491125, 10.996013446352867]
        assertSequenceAlmostEqual(self, result, expected, 7)

    def test_determined_sys_big(self):
        self.coef = np.array('1 1 1; 5 4 4; 4 5 2')
        self.rs = np.array('300; 1060; 1140')
        params = est.lse.recursive(self.coef, self.rs)
        result = np.dot(self.coef, params)
        result = [e[0, 0] for e in result]
        expected = [298.44086271560286, 1060.3673278413662, 1139.9639735518322]
        assertSequenceAlmostEqual(self, result, expected, 7)

    def test_overdetermined_sys(self):
        self.coef = np.array('2 1 5; 1 3 4; 0 5 -1; -1 2 3')
        self.rs = np.array('1; -7; -15; -8')
        params = est.lse.recursive(self.coef, self.rs)
        result = np.dot(self.coef, params)
        result = [e[0, 0] for e in result]
        expected = [0.9994501300038658, -6.99994998163175,
                    -14.999699950935845, -7.999050253967136]
        assertSequenceAlmostEqual(self, result, expected, 7)

