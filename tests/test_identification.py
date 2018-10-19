import unittest
from sample import identification as sid
from sample import data


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
        self.inp = [0, 1]
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (288, 2))

    def test_shape_double(self):
        self.setUp()
        self.fname = 'ipca2.dat'
        self.inp = [0, 1, 2]
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (287, 3))

    def test_shape_double_three(self):
        self.setUp()
        self.fname = 'ipca2.dat'
        self.inp = [0, 1, 2]
        self.order = 2
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (286, 6))

    # def test_armax_shape(self):
    #     self.setUp()
    #     self.fname = 'ipca1.dat'
    #     self.inp = [0, 1]
    #     with data.open_matrix('examples/' + self.fname) as dat:
    #         theta, _, _, _ = sid.idarmaxr(dat, self.order, self.delay)
    #         self.assertEqual(theta.shape, (3, 1))

    # def test_armax_shape_double(self):
    #     self.setUp()
    #     self.fname = 'ipca2.dat'
    #     self.inp = [0, 1, 2]
    #     with data.open_matrix('examples/' + self.fname) as dat:
    #         theta, _, _, _ = sid.idarmaxr(dat, self.order, self.delay)
    #         self.assertEqual(theta.shape, (4, 1))

    # def test_armax_shape_triple(self):
    #     self.setUp()
    #     self.fname = 'ipca3.dat'
    #     self.inp = [0, 1, 2, 3]
    #     with data.open_matrix('examples/' + self.fname) as dat:
    #         theta, _, _, _ = sid.idarmaxr(dat, self.order, self.delay)
    #         self.assertEqual(theta.shape, (5, 1))
