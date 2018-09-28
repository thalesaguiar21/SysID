import unittest
import numpy as np
from sample import data


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.fname = 'data_test.dat'
        self.out = 1

    def test_rdots(self):
        self.setUp()
        expected = [141.31, 198.22, 0.64945, 282.96, 198.22, 282.96,
                    0.650154, 403.73, 282.96, 403.73, 0.637122, 581.49]
        with data.open_matrix('examples/' + self.fname, sep=' ') as dots:
            dots = np.append([], dots).tolist()
            self.assertSequenceEqual(expected, dots)

    def test_rdots_shape(self):
        self.setUp()
        with data.open_matrix('examples/' + self.fname, sep=' ') as dots:
            self.assertEqual(dots.shape, (3, 4))

    def test_rdots_nonefile(self):
        self.setUp()
        self.fname = None
        try:
            with data.open_matrix(self.fname, sep=' ') as dots:
                self.fail('Dots from non existing file {}'.format(dots))
        except TypeError:
            pass

    def test_rdots_zerofile(self):
        self.setUp()
        self.fname = ''
        try:
            with data.open_matrix(self.fname, sep=' ') as dots:
                self.fail('Dots from non existing file {}'.format(dots))
        except FileNotFoundError:
            pass

    def test_rdots_nofile(self):
        self.setUp()
        self.fname = 'filethatdoesnotexistinfolder.dat'
        try:
            with data.open_matrix(self.fname, sep=' ') as dots:
                self.fail('Dots from non existing file {}'.format(dots))
        except FileNotFoundError:
            pass
