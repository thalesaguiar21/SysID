""" This module contains a collection of methods to solve systems
through Least Square Estimation (LSE).
"""
import numpy as np

import sysid.utils as idutils


def solve_matricial(coefs, res):
    """ Solve the given system using pseudo inverse model

    Args:
        coefs (ndarray): The system's coeficients
        res (ndarray): The system's output

    Returns:
        The parameters that solve the given system
    """
    pinv = np.lingalg.inv(coefs.T @ coefs)
    theta = pinv @ coefs.T @ res
    return theta


def solve_recursive(coefs, res, forget=1.0, confidence=1000):
    """ Solve the given system with a recursive least square estimation.

    Args:
        coefs(ndarray): The coefficient matrix
        res (ndarray): The expected values
        forget (float): The forget rate, defaults to 1.0
        confidence (int): The inital confidence value, defaults to 1000

    Returns:
        An estimation of parameters theta, that psi * theta ~ y
    """
    forget, confidence = _fix_args(forget, confidence)
    _validate_args(coefs, res)
    n_eqs, n_vars = coefs.shape
    covariances = np.eye(n_vars) * confidence
    theta = np.zeros(n_vars)
    for k in range(n_eqs):
        coefs_k = np.array([coefs[k, :]]).T
        term = covariances @ coefs_k
        denom = coefs_k.T@term + forget
        gain = (term / denom).reshape(n_vars,)
        theta += gain * (res[k] - coefs_k.T@theta)
        part = covariances @ coefs_k @ coefs_k.T @ covariances
        covariances -= part / denom
        covariances *= 1.0 / forget
    return theta


def _fix_args(forget_rate, confidence):
    fg_rate = _clip_forget_rate(forget_rate)
    conf = 1 if confidence <= 0 else confidence
    return fg_rate, conf


def _clip_forget_rate(forget_rate):
    return idutils.clip(forget_rate, a_min=1e-4, a_max=1.0)


def _validate_args(coef, rs):
    _is_legal_param(coef, rs)


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
