from numpy import eye, dot, zeros, ndarray, matrix
from numpy.linalg import inv

''' This module has some estimation functions developed during the class
of introduction to the Identification of Systems
'''


def __is_legal_param(coef, rs):
    ''' Performs validations along the number of parameters and dimension
    of the given system.
    '''
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
    ''' Check whether the given input is a row matrix (1, )

    Parameters
    ----------
    rs : np array
        The object to be checked

    Raise
    -----
    ValueError when the object fails as a row matrix
    '''
    if rs is not None and not isinstance(rs, ndarray):
        raise ValueError('Invalid type of object ' + str(type(rs)))
    if rs is not None and rs.shape[1] != 1:
        raise ValueError('Result matrix must be a column matrix.')


def recursive_lse(coef, rs, conf=1000, ffac=1.0, noise=None):
    '''
    Compute the Least Square estimation of a system

    y(k) = psi(k)theta(k) + eta(k)

    Parameters
    ----------
    coef : np matrix
        The coefficient matrix
    rs : column matrix (N, 1)
        The expected result matrix
    conf : float, defaults to 1000
        The initital confidence of the covariance matrix
    ffac : float [1e-4, 1.0]
        The forgetting factor
    noise : column matrix
        The noise to be added to the system output

    Returns
    ------
    theta : np column matrix
        The identified parameters
    P : np matrix
        The final covariance matrix
    variation : np matrix
        The parameters variation after each iteration
    '''
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
    '''
    Compute the LSE with matricial operations, that is,
    AX = B, then
    X = (A'A)A'B

    Parameters
    ----------
    coef : matrix
        The coefficients matrix
    rs : column matrix
        The result matrix

    Returns
    -------
    theta : column matrix
        The values that solve the system
    res : column matrix
        The residues of identified outputs
    ypred : column matrix
        The identified outputs
    '''
    __is_column_vector(rs)
    __is_legal_param(coef, rs)
    pinv = inv(dot(coef.T, coef))
    theta = dot(dot(pinv, coef.T), rs)
    ypred = dot(coef, theta)
    res = rs - ypred
    return theta, res, ypred


def kalman_filter(A, B, Q, u, y):
    # propagation
    X = []
    P = []
    C = []
    R = .5
    Id = eye(len(u)) * 1000
    for t in range(len(u)):
        X = dot(A, X) + dot(B, u[t])
        P = dot(A, dot(P, A.T)) + Q
        # update
        num = dot(P, C.T)
        K = num / (dot(C, num) + R)
        X = X + dot(K, y[t] - dot(C, X))
        P = dot((Id - dot(K, C)), P)
    return X
