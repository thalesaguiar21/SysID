import unittest

from .context import sysid
from sysid import identification as sid


class TestIdentification(unittest.TestCase):

    def setUp(self):
        self.fname = 'ipca2.dat'
        self.inp = 0
        self.out = 1
        self.order = 1
        self.delay = 0

    def test_shape_single(self):
        self.fname = 'ipca1.dat'
        self.inp = [0, 1]
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (288, 2))

    def test_shape_double(self):
        self.fname = 'ipca2.dat'
        self.inp = [0, 1, 2]
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (287, 3))

    def test_shape_double_three(self):
        self.fname = 'ipca2.dat'
        self.inp = [0, 1, 2]
        self.order = 2
        with data.open_matrix('examples/' + self.fname) as dat:
            A, _ = sid.idarx(dat, self.order, self.delay)
            self.assertEqual(A.shape, (286, 6))

