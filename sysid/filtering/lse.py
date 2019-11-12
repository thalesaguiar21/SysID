""" This module contains a collection of methods to solve systems
through Least Square Estimation (LSE).
"""
import numpy as np

from sysid.utils import clip


class Matricial(_LSE):
    """ """

    def __init__(self, psi, y):
        super().__init__(psi, y)

    def solve(self):
        """ Solve the given system with a matricial least square estimation.

        Args:
            psi (ndarray): The coefficient matrix
            y (ndarray): The expected values

        Returns:
            An estimation of parameters theta, that psi * theta ~ y
        """
        psi_pseudo_inv = inv(dot(self.psi.T, self.psi))
        theta = dot(dot(psi_pseudo_inv, self.psi.T), self.y)
        return theta


class Recursive(_LSE):
    """ """
    MIN_CONFIDENCE = 0
    MAX_CONFIDENCE = 10000
    MIN_FORGET_RATE = 1e-4
    MAX_FORGET_RATE = 1.0

    def __init__(self, psi, y, forget_rate=1.0, initial_confidence=1000):
        super().__init__(psi, y)
        QTD_VARIABLES = psi.shape[1]
        initial_confidence = clip(
            initial_confidence,
            a_min=Recursive.MIN_CONFIDENCE,
            a_max=Recursive.MAX_CONFIDENCE)
        self.forget_rate = clip(
            forget_rate,
            a_min=Recursive.MIN_FORGET_RATE,
            a_max=Recursive.MAX_FORGET_RATE)
        self.covariance = eye(QTD_VARIABLES) * initial_confidence
        self.theta = zeros((QTD_VARIABLES, 1))

    def solve(self):
        """ Solve the given system with a recursive least square estimation.

        Args:
            psi (ndarray): The coefficient matrix
            y (ndarray): The expected values

        Returns:
            An estimation of parameters theta, that psi * theta ~ y
        """
        for k in range(self.psi.shape[0]):
            psik = self.psi[k, :].T
            term = dot(self.covariance, psik)
            gain = term / ((dot(psik.T, term)) + self.forget_rate)
            self.theta += dot(gain, self.y[k] - dot(psik.T, self.theta))
            self.covariance = (1.0 / self.forget_rate) * (
                self.covariance - dot(gain, dot(psik.T, self.covariance)))
        return self.theta


def _validate_args(coef, rs):
    _is_legal_param(coef, rs)
    _is_column_vector(rs)


def _is_legal_param(coef, rs):
    """ Performs validations along the number of parameters """
    if coef is None or (coef is not None and coef.size == 0):
        raise ValueError('Coefficient matrix is None or empty!')
    if rs is None or (rs is not None and rs.size == 0):
        raise ValueError('Result matrix is None or empty!')
    if coef.shape[0] < coef.shape[1]:
        raise ValueError('Cannot solve indetermined system!')
    if coef.shape[0] != rs.shape[0]:
        raise ValueError('Expected {}, got {} values in result matrix!'.format(
            coef.shape[1], rs.shape[0]))


def _is_column_vector(rs):
    """ Check whether the given input is a row matrix (1, )

    Args:
        rs (ndarray) : The object to be checked

    Raises:
        ValueError: when the object fails as a row matrix
    """
    if rs is not None and not isinstance(rs, ndarray):
        raise ValueError('Invalid type of object ' + str(type(rs)))
    if rs is not None and len(rs.shape) != 1:
        raise ValueError('Result matrix must be a column matrix.')
