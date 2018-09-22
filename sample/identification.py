from __future__ import print_function
from __future__ import division
from numpy import zeros, array, dot, vstack
from numpy.random import normal
from metrics import stdev
from estimation import recursive_lse, mat_lse
import pdb

''' This module has some estimation functions developed during the class
of introduction to the Identification of Systems
'''


def __valid_io(u, y):
    if u.size != y.size:
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


def identify_arx_params(u, y, order, delay):
    __validate_id(u, y, order, delay)
    u = array(u)
    y = array(y)
    min_points = 3 * order * delay
    if y.size < min_points:
        raise ValueError(
            'The length of u and y must be at least 3 * order * delay')
    n_equations = y.size - order - delay
    B = zeros((n_equations, 1))
    A = zeros((n_equations, 2 * order))
    for i in xrange(n_equations):
        for j in xrange(order):
            A[i, j] = y[i + order + delay - j, 0]
            A[i, j + order] = u[i + order - j, 0]
        B[i, 0] = y[i + order + delay, 0]
    return A, B


def identify_arx_params_miso(u, y, order, delay):
    __valid_order(order)
    __valid_delay(delay)
    u = array(u)
    y = array(y)
    min_points = 3 * order * delay
    n_inps = u.shape[1]
    pdb.set_trace()
    if y.size < min_points:
        raise ValueError(
            'The length of u and y must be at least 3 * order * delay')
    n_equations = y.size - order - delay
    B = zeros((n_equations, 1))
    A = zeros((n_equations, 2 * (order + n_inps - 1)))
    for i in xrange(n_equations):
        for j in xrange(order):
            A[i, j] = y[i + order + delay - j, 0]
            for k in xrange(n_inps):
                A[i, j + order * (k + 1)] = u[i + order - j, k]
        B[i, 0] = y[i + order + delay, 0]
    return A, B


def identify_arx(u, y, order, delay):
    ''' Identify the structural parameters of the ARX system

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
        The residue of the estimated outputs
    ypred : numpy column matrix
        The predicted output
    '''
    A, B = identify_arx_params_miso(u, y, order, delay)
    return mat_lse(A, B)


def identify_arx_rec(u, y, order, delay, conf=1000, ffactor=1.0):
    ''' Identify the structural parameters of the ARX system

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
    conf : float, defaults to 1000
        The initial confidence value for covariance matrix
    ffactor : float, defaults to 1.0
        The forgetting factor of LSE

    Returns
    -------
    theta : numpy column matrix
        The identified parameters
    res : numpy column matrix
        The residue of the estimated output
    ypred : numpy column matrix
        The predicted output
    phist : matrix
        The parameters variation along the samples
    '''
    A, B = identify_arx_params_miso(u, y, order, delay)
    theta, phist = recursive_lse(A, B, conf, ffactor)
    ypred = dot(A, theta)
    res = B - ypred
    return theta, res, ypred, phist


def __identify_armax_int_params(u, y, order, delay, e):
    __validate_id(u, y, order, delay)
    if y.size != e.size:
        raise ValueError('The length of e and y must be the same!')

    n_points = y.size
    n_min_points = 4 * order + delay
    if n_points < n_min_points:
        raise ValueError(
            'The minimum size of u, e and y must be 4 * order + delay')

    n_equation = n_points - order - delay
    B = zeros((n_equation, 1))
    A = zeros((n_equation, 3 * order))
    for i in xrange(n_equation):
        for j in xrange(order):
            A[i, j] = y[i + order + delay - j, 0]
            A[i, j + order] = u[i + order - j, 0]
            A[i, j + 2 * order] = e[i + order + delay - j, 0]
        B[i, 0] = y[i + order + delay, 0]
    return A, B


def __identify_armax_int(u, y, order, delay, e):
    A, B = __identify_armax_int_params(u, y, order, delay, e)
    return mat_lse(A, B)


def __identify_armax_int_rec(u, y, order, delay, e, conf, ffactor):
    A, B = __identify_armax_int_params(u, y, order, delay, e)
    theta, phist = recursive_lse(A, B, conf, ffactor)
    ypred = dot(A, theta)
    res = B - ypred
    return theta, res, ypred, phist


def identify_armax_rec(u, y, order, delay, conf=1000, ffac=1.0):
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
    '''
    theta, res, ypred = identify_arx(u, y, order, delay)
    stdev_res = stdev(res)
    stdev_res_ant = 2.0 * stdev_res
    N = 0
    phist = []
    while abs(stdev_res_ant - stdev_res) / stdev_res > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        theta, res, ypred, phist = __identify_armax_int_rec(
            u, y, order, delay, e_estim, conf, ffac)
        stdev_res_ant = stdev_res
        stdev_res = stdev(res)
        N = N + 1
    return theta, res, ypred, phist


def identify_armax(u, y, order, delay):
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
    '''
    theta, res, _ = identify_arx(u, y, order, delay)
    stdev_res = stdev(res)
    stdev_res_ant = 2.0 * stdev_res
    N = 0
    while abs(stdev_res_ant - stdev_res) / stdev_res > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        theta, res, ypred = __identify_armax_int(u, y, order, delay, e_estim)
        stdev_res_ant = stdev_res
        stdev_res = stdev(res)
        N = N + 1
    return theta, res, ypred
