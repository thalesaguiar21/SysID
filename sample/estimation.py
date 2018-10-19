from numpy import eye, dot, zeros, ndarray, matrix, append
from numpy.linalg import inv
import pdb

""" This module has some estimation functions developed during the class
of introduction to the Identification of Systems.
"""


_MIN_CONFIDENCE = 0
_MAX_FORGET_FACTOR = 1.0
_MIN_FORGET_FACTOR = 1e-4


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
    conf = _MIN_CONFIDENCE if conf < _MIN_CONFIDENCE else conf
    ffac = _MIN_FORGET_FACTOR if ffac < _MIN_FORGET_FACTOR else ffac
    ffac = _MAX_FORGET_FACTOR if ffac > _MAX_FORGET_FACTOR else ffac
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


class KalmanFilter(object):
    """
    """

    def __init__(self, propag, entry, observation, icovariance=0.1):
        """ Initialize the prediction model for Kalman Filter

        Args:
            propag (ndarray): The propagation matrix
            entry (ndarray): The entry matrix
            observation (ndarray): The observation matrix
            icovariance (float, optional): The initial states covariances. Defa
                ults to 0.1.
        """
        self._statehist = []
        self._covariancehist = []
        self._QTD_STATE = propag.shape[0]
        self.propag = propag
        self.entry = entry
        self.observation = observation
        self.states = zeros(self._QTD_STATE)
        self.covariances = eye(self._QTD_STATE) * icovariance

    def filtrate(self, measures, inputs, dyn_noise=None, measurenoise=None):
        """ Filtrate the measures with the given prediction model

        Args:
            measures (ndarray): The sensors measurements
            inputs (ndarray): The exogenous inputs
            dyn_noise (ndarray, optional): The dynamic noise matrix. Defaults
                to M = [0] with order NxN, where N is the number of states.
            measurenoise (ndarray, optional): The measurement noise matrix. Def
                aults to M = [0] with order NxN, where N is the number of state
                s.

        Returns:
            The states and covariances history
        """
        measurenoise = self._init_noise(measurenoise)
        dyn_noise = self._init_noise(dyn_noise)
        self._clean_history()

        for t in range(len(measures)):
            self._propagate(dyn_noise, inputs)
            self._adjust(measurenoise, measures[t])
            self._append_estimation()
        return self._statehist, self._covariancehist

    def _clean_history(self):
        self._statehist = []
        self._covariancehist = []

    def _init_noise(self, noise):
        if noise is None:
            return zeros(self._QTD_STATE)

    def _propagate(self, dyn_noise, inputs):
        self.states = dot(self.propag, self.states) + dot(self.entry, inputs)
        self.covariances = dot(
            self.propag, dot(self.covariances, self.propag.T))
        self.covariances = self.covariances + dyn_noise

    def _adjust(self, measurenoise, measures):
        """ States and covariances are changed inside """
        term = dot(self.covariances, self.observation.T)
        gain = dot(term, inv(dot(self.observation, term) + measurenoise))
        self.states += dot(
            gain, measures.T - dot(self.observation, self.states))
        self.covariances -= dot(dot(gain, self.observation), self.covariances)

    def _append_estimation(self, state, covariance):
        self._statehist.append(append([], state.T))
        for i in range(self._QTD_STATE):
            self._covariancehist.append(covariance[i, i])
