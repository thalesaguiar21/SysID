import unittest
import numpy as np
from sample import data


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.fname = 'data_test.dat'
        self.out = 1

    def __format_inout(self, inp, out):
        inaux = np.append([], inp)
        outaux = np.append([], out)
        return inaux, outaux

    def test_scalar_inp(self):
        self.setUp()
        self.inp = [0, 1]
        einp = [141.31, 198.22, 282.96]
        eout = [198.22, 282.96, 403.73]
        inp, out = data.r_dots(self.fname, self.inp)
        inp, out = self.__format_inout(inp, out)
        self.assertSequenceEqual(einp, inp.tolist())
        self.assertSequenceEqual(eout, out.tolist())

    def test_empty_inp(self):
        self.setUp()
        try:
            data.r_dots(self.fname, [])
            self.fail()
        except ValueError:
            pass

    def test_none_inp(self):
        self.setUp()
        try:
            data.r_dots(self.fname, None)
            self.fail()
        except ValueError:
            pass

    def test_inp_offbound(self):
        self.setUp()
        try:
            data.r_dots(self.fname, [0, 1000, 4])
            self.fail()
        except IndexError:
            pass

    def test_inp_negative(self):
        self.setUp()
        try:
            data.r_dots(self.fname, [0, -1, -400])
            self.fail()
        except ValueError:
            pass

    def test_array_inp(self):
        self.setUp()
        self.inp = [0, 2, 1]
        einp = [[141.31, 0.64945],
                [198.22, 0.650154],
                [282.96, 0.637122]]
        eout = [198.22, 282.96, 403.73]
        inp, out = data.r_dots(self.fname, self.inp)
        inp, out = self.__format_inout(inp, out)
        einp = np.append([], einp)
        self.assertSequenceEqual(einp.tolist(), inp.tolist())
        self.assertSequenceEqual(eout, out.tolist())

    def test_none_out(self):
        self.setUp()
        self.inp = [0, 1, None]
        try:
            data.r_dots(self.fname, self.inp)
            self.fail()
        except ValueError:
            pass

    def test_out_offbound(self):
        self.setUp()
        self.inp = [0, 1, 4]
        try:
            data.r_dots(self.fname, self.inp)
            self.fail()
        except IndexError:
            pass

    def test_negative_out(self):
        self.setUp()
        self.inp = [0, 1, -1]
        try:
            data.r_dots(self.fname, self.inp)
            self.fail()
        except ValueError:
            pass

    def test_out_farbound(self):
        self.setUp()
        self.inp = [0, 1, 1000]
        try:
            data.r_dots(self.fname, self.inp)
            self.fail()
        except IndexError:
            pass

    def test_shapes(self):
        self.setUp()
        self.inp = [0, 1, 1]
        inp, out = data.r_dots(self.fname, self.inp)
        self.assertEqual(inp.shape, (3, 2))
        self.assertEqual(out.shape, (3, 1))

    def test_rdots(self):
        self.setUp()
        expected = [141.31, 198.22, 0.64945, 282.96, 198.22, 282.96,
                    0.650154, 403.73, 282.96, 403.73, 0.637122, 581.49]
        dots = data.readdots(self.fname, sep=' ')
        dots = np.append([], dots).tolist()
        self.assertSequenceEqual(expected, dots)

    def test_rdots_shape(self):
        self.setUp()
        dots = data.readdots(self.fname, sep=' ')
        self.assertEqual(dots.shape, (3, 4))

    def test_rdots_nonefile(self):
        self.setUp()
        self.fname = None
        try:
            dots = data.readdots(self.fname, sep=' ')
            self.fail('Dots from non existing file {}', dots)
        except ValueError:
            pass

    def test_rdots_zerofile(self):
        self.setUp()
        self.fname = ''
        try:
            dots = data.readdots(self.fname, sep=' ')
            self.fail('Dots from non existing file {}', dots)
        except ValueError:
            pass

    def test_rdots_nofile(self):
        self.setUp()
        self.fname = 'filethatdoesnotexistinfolder.dat'
        try:
            dots = data.readdots(self.fname, sep=' ')
            self.fail('Dots from non existing file {}', dots)
        except FileNotFoundError:
            pass
