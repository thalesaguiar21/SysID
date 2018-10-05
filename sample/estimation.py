from numpy import eye, dot, zeros, ndarray, matrix
from numpy.linalg import inv

""" This module has some estimation functions developed during the class
of introduction to the Identification of Systems.
"""


def __is_legal_param(coef, rs):
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


def __is_column_vector(rs):
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


def recursive_lse(coef, rs, conf=1000, ffac=1.0, noise=None):
    """ Compute the Least Square estimation of a system
    K = (P - Psi) / Psi' x P x Psi
    Theta = Theta + K[y - Psi' x Theta]
    P =  P - K x Psi' x P

    Args:
        coef (ndarray): The coefficient matrix
        rs (ndarray): The expected result matrix
        conf (float): Confidence of the covariance matrix. Defaults to 1000
        ffac (float): The forgetting factor. Defaults to 1.0
        noise (ndarray): The noise to be added to the system output

    Returns:
        theta : the estimated aprameters
        variation : paramaters gain over time
    """
    conf = 0 if conf < 0 else conf
    ffac = 1e-4 if ffac < 1e-4 else ffac
    ffac = 1.0 if ffac > 1.0 else ffac
    __is_column_vector(rs)
    __is_legal_param(coef, rs)

    psi = matrix(coef)
    P = eye(psi.shape[1]) * conf
    theta = zeros((psi.shape[1], 1))
    noise = zeros((psi.shape[0], 1)) if noise is None else noise
    Y = matrix(rs)
    params_variation = []

    for k in range(psi.shape[0]):
        psi_k = psi[k, :].T  # To keep notation similar to math formulae
        K = dot(P, psi_k) / ((dot(psi_k.T, dot(P, psi_k))) + ffac)
        params_variation.append([theta[i, 0] for i in range(theta.shape[0])])
        theta = theta + dot(K, (Y[k] - dot(psi_k.T, theta)))
        P = (1.0 / ffac) * (P - dot(K, dot(psi_k.T, P)))

    return theta, params_variation


def mat_lse(coef, rs):
    """ Compute the LSE with matricial operations, that is,
    AX = B, then
    X = (A'A)A'B

    Args:
        coef (ndarray): The coefficients matrix
        rs (ndarray): The result matrix

    Returns:
        theta: The estimated parameters
        res: The residues of identified outputs
        ypred: The identified outputs
    """
    __is_column_vector(rs)
    __is_legal_param(coef, rs)
    pinv = inv(dot(coef.T, coef))
    theta = dot(dot(pinv, coef.T), rs)
    ypred = dot(coef, theta)
    res = rs - ypred
    return theta, res, ypred
