from numpy import zeros, vstack, dot
from numpy.random import normal
from sysid.metrics import stdev
from sysid.filtering import lse
from sysid.utils import calc_residues
from enum import Enum

""" This module has some identiication functions developed during the class
of introduction to the Identification of Systems.
"""


class Structure(Enum):
    """ Supported system structures """
    ARX = 'arx'
    ARMAX = 'armax'
    SIMPLE_CAR = 'scar'

    def __str__(self):
        return '{0}'.format(self.value)


class Stage(Enum):
    """ Supported identification steps """
    TRAINING = 'train'
    VALIDATING = 'val'

    def __str__(self):
        return '{0}'.format(self.value)


class Noise(object):
    """ Supported noise types """
    WHITE = 1


def _get_io(data):
    """ Separate the last columns as output and the rest as inputs """
    return data[:, :-1], data[:, -1:]


def _valid_io(u, y):
    if u.shape[0] != y.size:
        raise ValueError('Number of input/outputs must be equal!')


def _valid_delay(delay):
    if delay < 0:
        raise ValueError('Delay ust be positive!')


def _valid_order(order):
    if order < 1:
        raise ValueError('Order must be a non null positive number!')


def _validate_id(u, y, order, delay):
    _valid_io(u, y)
    _valid_delay(delay)
    _valid_order(order)


def _add_noise(output, sdev, noise=Noise.WHITE):
    """ Add noise to the given output of the system

    Args:
        output (ndarray): The clean output
        stdev (float): The standard deviation of the noise
        noise (Noise): The type of noise. Defaults to WHITE

    Returns:
        y_noise (ndarray): The output of the system with added noise
    """
    if noise == Noise.WHITE:
        return output + normal(0, sdev, (output.size, 1))
    else:
        raise ValueError('Invalid type of noise: {}'.format(noise))


def idarx(data, order, delay):
    """ Compute the regression matrix and the expected values

    Args:
        data (ndarray): The input/output matrix
        order (int): The system order
        delay (int): The system delay

    Returns:
        A (2D list): The regression matrix
        B (2D list): The expected values
    """
    _valid_order(order)
    _valid_delay(delay)
    inp, out = _get_io(data)
    # pdb.set_trace()
    min_points = 3 * order * delay
    n_inps = inp.shape[1]
    if out.size < min_points:
        raise ValueError(
            'The length of u and y must be at least 3 * order * delay')
    n_equations = out.size - order - delay
    B = zeros((n_equations, 1))
    A = zeros((n_equations, order * (n_inps + 1)))
    for i in range(n_equations):
        for j in range(order):
            A[i, j] = out[i + order + delay - j, 0]
            for k in range(n_inps):
                A[i, j + order * (k + 1)] = inp[i + order - j, k]
        B[i, 0] = out[i + order + delay, 0]
    return A, B


def _idarmax_p(u, y, order, delay, e):
    """ Auxiliary function to generate the system's regressors """
    _validate_id(u, y, order, delay)
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


def _initarmax(dat, order, delay):
    """ Auxiliary function to initialize some variables for ARMAX id """
    A, B = idarx(dat, order, delay)
    theta = lse.recursive(A, B)
    ypred = dot(A, theta)
    res = B - ypred
    sdev = stdev(res)
    return res, sdev, 2 * sdev


def idarmax(dat, order, delay):
    """ Identify the structural parameters of the ARMAX system

    Args:
        dat (ndarray): Inputs and outputs of the system
        order (int): Order of the system
        delay (int): The delay of the system

    Returns:
        theta (ndarray): The identified parameters
        res (ndarray): The residues of the identified parameters theta
        ypred (ndarray): The predicted outputs
    """
    inp, out = _get_io(dat)
    N = 0
    res, sdev, oldsdev = _initarmax(dat, order, delay)
    while abs(oldsdev - sdev) / sdev > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        A, B = _idarmax_p(inp, out, order, delay, e_estim)
        theta = lse.matricial(A, B)
        res = calc_residues(A, B, theta)
        ypred = dot(A, theta)
        oldsdev = sdev
        sdev = stdev(res)
        N = N + 1
    return theta, res, ypred


def idarmaxr(dat, order, delay, conf=1000, forg=1.0):
    """ Identify the structural parameters of the ARMAX system with a
    recursive strategy

    Args:
        u (ndarray): System exogenous inputs
        y (ndarray): Measured outputs of the system
        order (int): Order of the system
        delay (int): The delay of the system

    Returns:
        theta (ndarray): The identified parameters
        res (ndarray): The residues of the identified parameters theta
        ypred (ndarray): The predicted outputs
        phist (2D array): The paramater variation along samples
    """
    lse.initial_confidence = conf
    lse.forget_rate = forg
    N = 0
    u, y = _get_io(dat)
    res, sdev, oldsdev = _initarmax(dat, order, delay)
    while abs(oldsdev - sdev) / sdev > 0.01 and N < 30:
        e_estim = vstack((zeros((order + delay, 1)), res))
        A, B = _idarmax_p(u, y, order, delay, e_estim)
        theta, phist = lse.recursive(A, B)
        ypred = dot(A, theta)
        res = B - ypred
        oldsdev = sdev
        sdev = stdev(res)
        N = N + 1
    return theta, res, ypred, phist
