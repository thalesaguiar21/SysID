""" This module contains a collection of methods to solve systems
through Least Square Estimation (LSE).
"""
from numpy import eye, dot, zeros, ndarray
from numpy.linalg import inv
from sample.estimation.utils import clamp

_MIN_CONFIDENCE = 0
_MAX_CONFIDENCE = 10000
_MIN_FORGET_RATE = 1e-4
_MAX_FORGET_RATE = 1.0
initial_confidence = 1000
forget_rate = 1.0


def recursive(psi, y):
    """ Solve the given system with a recursive least square estimation.

    Args:
        psi (ndarray): The coefficient matrix
        y (ndarray): The expected values

    Returns:
        An estimation of parameters theta, that psi * theta ~ y
    """
    _validate_args(psi, y)
    QTD_VARIABLES = psi.shape[1]
    forget_rate, initial_confidence = _clamp_rates()
    cov = eye(QTD_VARIABLES) * initial_confidence
    theta = zeros((QTD_VARIABLES, 1))
    for k in range(psi.shape[0]):
        psik = psi[k, :].T
        gain = dot(cov, psik) / ((dot(psik.T, dot(cov, psik))) + forget_rate)
        theta += dot(gain, y[k] - dot(psik.T, theta))
        cov = (1.0 / forget_rate) * (cov - dot(gain, dot(psik.T, cov)))
    return theta


def matricial(psi, y):
    """ Solve the given system with a matricial least square estimation.

    Args:
        psi (ndarray): The coefficient matrix
        y (ndarray): The expected values

    Returns:
        An estimation of parameters theta, that psi * theta ~ y
    """
    psi_pseudo_inv = inv(dot(psi.T, psi))
    theta = dot(dot(psi_pseudo_inv, psi.T), y)
    return theta


def _clamp_rates():
    clampedforget = clamp(forget_rate, _MIN_FORGET_RATE, _MAX_FORGET_RATE)
    clampedconf = clamp(initial_confidence, _MIN_CONFIDENCE, _MAX_CONFIDENCE)
    return clampedforget, clampedconf


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
    if rs is not None and rs.shape[1] != 1:
        raise ValueError('Result matrix must be a column matrix.')
