import unittest2 as unittest
import numpy as np
from context import dut
import pdb


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.fname = 'data_test.dat'
        self.inp = 0
        self.out = 1

    def __format_inout(self, inp, out):
        inaux = np.append([], inp)
        outaux = np.append([], out)
        return inaux, outaux

    def test_scalar_inp(self):
        self.setUp()
        self.inp = 0
        einp = [141.31, 198.22, 282.96]
        eout = [198.22, 282.96, 403.73]
        inp, out = dut.r_dots(self.fname, self.inp, self.out)
        inp, out = self.__format_inout(inp, out)
        self.assertSequenceEqual(einp, inp.tolist())
        self.assertSequenceEqual(eout, out.tolist())

    def test_empty_inp(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, [], 1)
            self.fail()
        except ValueError:
            pass

    def test_none_inp(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, None, 1)
            self.fail()
        except ValueError:
            pass

    def test_scalar_neg_inp(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, -290, self.out)
            self.fail()
        except ValueError:
            pass

    def test_inp_offbound(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, [0, 1000, 4], self.out)
            self.fail()
        except IndexError:
            pass

    def test_inp_negative(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, [0, -1, -400], self.out)
            self.fail()
        except ValueError:
            pass

    def test_array_inp(self):
        self.setUp()
        self.inp = [0, 2]
        einp = [[141.31, 0.64945],
                [198.22, 0.650154],
                [282.96, 0.637122]]
        eout = [198.22, 282.96, 403.73]
        inp, out = dut.r_dots(self.fname, self.inp, self.out)
        inp, out = self.__format_inout(inp, out)
        einp = np.append([], einp)
        self.assertSequenceEqual(einp.tolist(), inp.tolist())
        self.assertSequenceEqual(eout, out.tolist())

    def test_none_out(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, self.inp, None)
            self.fail()
        except ValueError:
            pass

    def test_out_offbound(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, self.inp, 4)
            self.fail()
        except IndexError:
            pass

    def test_negative_out(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, self.inp, -1)
            self.fail()
        except ValueError:
            pass

    def test_out_farbound(self):
        self.setUp()
        try:
            dut.r_dots(self.fname, self.inp, 1000)
            self.fail()
        except IndexError:
            pass
