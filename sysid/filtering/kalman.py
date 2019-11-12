from numpy import zeros, dot, eye, append, transpose, diag, array
from numpy.random import rand
from numpy.linalg import inv
from sample.metrics import variance
# from math import sqrt
# import pdb


class Linear(object):
    """ """

    def __init__(self, propag, entry, observation, dyn_cov, measure_cov):
        self.propag = propag
        self.entry = entry
        self.observation = observation
        self.QTD_STATES = propag.shape[0]
        self.QTD_OBS = observation.shape[0]
        self.states = rand(self.QTD_STATES, 1)
        self.covariances = zeros((self.QTD_STATES, self.QTD_STATES))
        self.measure_noise = _initial_measure_cov(measure_cov, self.QTD_OBS)
        self.dyn_noise = eye(self.QTD_STATES) * dyn_cov
        self._statehist = []
        self._covariancehist = []

    def filtrate(self, measures, inputs):
        self._clear_history()
        _filtrate(self, measures, inputs)
        return self._statehist, self._covariancehist

    def append_history(self):
        if self.states is not None and self.covariances is not None:
            self._statehist.append(append([], self.states.T))
            self._covariancehist.append(_get_main_diag2D(self.covariances))

    def _clear_history(self):
        self._statehist.clear()
        self._covariancehist.clear()

    def __str__(self):
        tmpl = ['Propag\n{}\n\n', 'Observation\n{}\n\n', 'Entry\n{}\n\n']
        descr = ''.join(tmpl).format(self.propag, self.observation, self.entry)
        return descr


class Extended(Linear):
    """ EKF """

    def smooth(self, measures, inputs):
        self._update_measure_noise(measures)
        self._update_dyn_noise()
        self.filtrate(measures, inputs)
        return self._statehist, self._covariancehist

    def _update_dyn_noise(self):
        residues = self._compute_states_residues()
        for state in range(self.QTD_STATES):
            self.dyn_noise[state, state] = variance(residues[state])

    def _compute_states_residues(self):
        residues = []
        states = array(self._statehist)
        for j in range(1, len(states)):
            residues.append(states[j] - dot(self.propag, states[j - 1]))
        return array(residues).T

    def _update_measure_noise(self, measures):
        states = array(self._statehist)
        residues = measures.T - dot(self.observation, states.T)
        for observ in range(self.QTD_OBS):
            self.measure_noise[observ, observ] = variance(residues[observ])

    def noises(self):
        return self.dyn_noise, self.measure_noise


class Unscented(Linear):
    """ """

    def __init__(self):
        self.sigma_points = []
        self.sigma_weights = []


def _calculate_sigma_points(dimension, mean, lambda_, covariances):
    pass


def _sigma_weights(dimension, lambda_, alpha, beta):
    pass


def _initial_measure_cov(cov, nobserv):
    # The float cast prevents atribuitions to be autocast to int
    return diag([float(cov) ** 2 for i in range(nobserv)])


def _get_main_diag2D(matrix):
    return [matrix[i, i] for i in range(len(matrix))]


def _filtrate(model, measures, inputs):
    """ Estimate the states and covariances for the kalman model. The propagati
    on, entry and observation matrices must be given before calling this method

    Args:
        measures (ndarray): The sensor measurements
        inputs (ndarray): The exogenous inputs

    Returns:
        The states and covariances through time
    """
    for t in range(len(measures)):
        _propagate(model, inputs)
        _adjust(model, measures[t])
        model.append_history()


def _propagate(model, inputs):
    """ Propagate the states and covariances to the next instant in time """
    model.states = dot(model.propag, model.states) + dot(model.entry, inputs)
    model.covariances = dot(
        model.propag, dot(model.covariances, model.propag.T))
    model.covariances += model.dyn_noise


def _adjust(model, measures):
    """ Adjust the predictions with the measurementes """
    term = dot(model.covariances, model.observation.T)
    gain = dot(term, inv(dot(model.observation, term) + model.measure_noise))
    model.states += dot(
        gain, transpose([measures]) - dot(model.observation, model.states)
    )
    state_gain = eye(model.QTD_STATES) - dot(gain, model.observation)
    model.covariances = dot(state_gain, model.covariances)
