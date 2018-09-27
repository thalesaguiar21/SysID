from __future__ import print_function
from __future__ import division
from numpy import zeros, array, vstack, dot
from numpy.random import normal
from sample.metrics import stdev
from sample.estimation import recursive_lse, mat_lse
# import pdb

''' This module has some estimation functions developed during the class
of introduction to the Identification of Systems
'''


def __valid_io(u, y):
    if u.shape[0] != y.size:
        raise ValueError('Number of input/outputs must be equal!')


def __valid_delay(delay):
    if delay < 0:
        raise ValueError('Delay ust be positive!')


def __valid_order(order):
    if order < 1:
        raise ValueError('Order must be a non null positive number!')


def __validate_id(u, y, order, delay):
    __valid_io(u, y)
    __valid_delay(delay)
    __valid_order(order)


def __add_noise(output, sdev, noise='gauss'):
    ''' Add noise to the given output of the system

    Parameters
    ----------
    output : numpy column matrix
        The clean output
    stdev : float
        The standard deviation of the noise
    noise : str, defaults to 'gauss'
        The type of noise

    Returns
    -------
    y_noise : numpy column matrix
        The output of the system with added noise
    '''
    if noise == 'gauss':
        return output + normal(0, sdev, (output.size, 1))
    else:
        raise ValueError('Invalid type of noise: {}'.format(noise))


def idarx(u, y, order, delay):
    ''' Compute the regression matrix and the expected values

    Parameters
    ----------
    u : numpy matrix
        The inputs
    y : column matrix
        The outputs
    order : non null positive int
        The system order
    delay : positive int
        The system delay

    Returns
    -------
    A : matrix
        The regression matrix
    B : matrix
        The expected values
    '''
    __valid_order(order)
    __valid_delay(delay)
    u = array(u)
    y = array(y)
    min_points = 3 * order * delay
    n_inps = u.shape[1]
    if y.size < min_points:
        raise ValueError(
            'The length of u and y must be at least 3 * order * delay')
    n_equations = y.size - order - delay
    B = zeros((n_equations, 1))
    A = zeros((n_equations, order * (n_inps + 1)))
    for i in range(n_equations):
        for j in range(order):
            A[i, j] = y[i + order + delay - j, 0]
            for k in range(n_inps):
                A[i, j + order * (k + 1)] = u[i + order - j, k]
        B[i, 0] = y[i + order + delay, 0]
    return A, B


def __idarmax_p(u, y, order, delay, e):
    ''' Auxiliary function to generate the system's regressors '''
    __validate_id(u, y, order, delay)
    if y.size != e.size:
        raise ValueError('The length of e and y must be the same!')

    n_points = y.size
    n_min_points = 4 * order + delay
    if n_points < n_min_points:
        raise ValueError(
            'The minimum size of u, e and y must be 4 * order + delay')

    n_inps = u.shape[1]
    n_equation = n_points - order - delay
    B = zeros((n_equation, 1))
    A = zeros((n_equation, order * (n_inps + 2)))
    for i in range(n_equation):
        for j in range(order):
            A[i, j] = y[i + order + delay - j, 0]
            for k in range(n_inps):
                A[i, j + order * (k + 1)] = u[i + order - j, k]
            A[i, j + n_inps * order] = e[i + order + delay - j, 0]
        B[i, 0] = y[i + order + delay, 0]
    return A, B


def __initarmax(u, y, order, delay):
    ''' Auxiliary functino to initialize some variables for ARMAX id '''
    A, B = idarx(u, y, order, delay)
    theta, _ = recursive_lse(A, B, 1000, 1.0)
    ypred = dot(A, theta)
    res = B - ypred
    sdev = stdev(res)
    return res, sdev, 2 * sdev


def idarmax(u, y, order, delay):
    ''' Identify the structural parameters of the ARMAX system

    Parameters
    ----------
    u : numpy column matrix (n, 1)
        Inputs of the system
    y : numpy column matrix (n, 1)
        Outputs of the system
    order : int
        Order of the system
    delay : int
        The delay of the system

    Returns
    -------
    theta : numpy column matrix
        The identified parameters
    res : numpy column matrix
        The residues of the identified parameters theta
    ypred : numpy column matrix
        The predicted outputs
    '''
    N = 0
    res, sdev, oldsdev = __initarmax(u, y, order, delay)
    while abs(oldsdev - sdev) / sdev > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        A, B = __idarmax_p(u, y, order, delay, e_estim)
        theta, res, ypred = mat_lse(A, B)
        oldsdev = sdev
        sdev = stdev(res)
        N = N + 1
    return theta, res, ypred


def idarmaxr(u, y, order, delay, conf=1000, forg=1.0):
    ''' Identify the structural parameters of the ARMAX system with a
        recursive strategy

    Parameters
    ----------
    u : numpy column matrix (n, 1)
        Inputs of the system
    y : numpy column matrix (n, 1)
        Outputs of the system
    order : int
        Order of the system
    delay : int
        The delay of the system

    Returns
    -------
    theta : numpy column matrix
        The identified parameters
    res : numpy column matrix
        The residues of the identified parameters theta
    ypred : numpy column matrix
        The predicted outputs
    phist : matrix
        The paramater variation along samples
    '''
    N = 0
    res, sdev, oldsdev = __initarmax(u, y, order, delay)
    while abs(oldsdev - sdev) / sdev > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        A, B = __idarmax_p(u, y, order, delay, e_estim)
        theta, phist = recursive_lse(A, B, conf, forg)
        ypred = dot(A, theta)
        res = B - ypred
        oldsdev = sdev
        sdev = stdev(res)
        N = N + 1
    return theta, res, ypred, phist
