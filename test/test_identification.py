import unittest2 as unittest
from context import identification as sid
from context import dut


class TestIdentification(unittest.TestCase):

    def setUp(self):
        self.fname = 'ipca2.dat'
        self.inp = 0
        self.out = 1
        self.order = 1
        self.delay = 0

    def test_shape_single(self):
        self.setUp()
        self.fname = 'ipca1.dat'
        self.inp = 0
        self.out = 1
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        A, _ = sid.identify_arx_params(u, y, self.order, self.delay)
        self.assertEqual(A.shape, (288, 2))

    def test_shape_double(self):
        self.setUp()
        self.fname = 'ipca2.dat'
        self.inp = [0, 1]
        self.out = 2
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        A, _ = sid.identify_arx_params(u, y, self.order, self.delay)
        self.assertEqual(A.shape, (287, 3))

    def test_shape_double_three(self):
        self.setUp()
        self.fname = 'ipca2.dat'
        self.inp = [0, 1]
        self.out = 2
        self.order = 2
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        A, _ = sid.identify_arx_params(u, y, self.order, self.delay)
        self.assertEqual(A.shape, (286, 6))

    def test_armax_shape(self):
        self.setUp()
        self.fname = 'ipca1.dat'
        self.inp = 0
        self.out = 1
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        theta, _, _, _ = sid.identify_armax_rec(u, y, self.order, self.delay)
        self.assertEqual(theta.shape, (3, 1))

    def test_armax_shape_double(self):
        self.setUp()
        self.fname = 'ipca2.dat'
        self.inp = [0, 1]
        self.out = 2
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        theta, _, _, _ = sid.identify_armax_rec(u, y, self.order, self.delay)
        self.assertEqual(theta.shape, (4, 1))

    def test_armax_shape_triple(self):
        self.setUp()
        self.fname = 'ipca3.dat'
        self.inp = [0, 1, 2]
        self.out = 3
        u, y = dut.r_dots(self.fname, self.inp, self.out, sep='\t')
        theta, _, _, _ = sid.identify_armax_rec(u, y, self.order, self.delay)
        self.assertEqual(theta.shape, (5, 1))
