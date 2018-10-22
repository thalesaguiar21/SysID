from numpy import zeros, dot, eye, append, transpose, diag
from numpy.linalg import inv
# import pdb

initital_covariance = 1000
propag = None
entry = None
observation = None
noise_update_rate = 10
_statehist = []
_covariancehist = []
_dyn_noise = diag([0.25, 2.25, 6.25, 12.25, 20.25])
_measurenoise = diag([25, 25])
_QTD_STATES = 0


def filtrate(measures, inputs):
    """ Estimate the states and covariances for the kalman model. The propagati
    on, entry and observation matrices must be given before calling this method

    Args:
        measures (ndarray): The sensor measurements
        inputs (ndarray): The exogenous inputs

    Returns:
        The states and covariances through time
    """
    _QTD_STATES = propag.shape[0]
    states = zeros((_QTD_STATES, 1))
    covariances = eye(_QTD_STATES) * initital_covariance
    for t in range(len(measures)):
        states, covariances = _propagate(states, covariances, inputs)
        states, covariances = _adjust(states, covariances, measures[t])
        _append_history(states, covariances)
    return _statehist, _covariancehist


def _propagate(states, covariances, inputs):
    """ Propagate the states and covariances to the next instant in time """
    states = dot(propag, states) + dot(entry, inputs)
    covariances = dot(propag, dot(covariances, propag.T)) + _dyn_noise
    return states, covariances


def _adjust(states, covariances, measures):
    """ Adjust the predictions with the measurementes """
    term = dot(covariances, observation.T)
    gain = dot(term, inv(dot(observation, term) + _measurenoise))
    states = states + dot(
        gain, transpose([measures]) - dot(observation, states)
    )
    covariances = covariances - dot(dot(gain, observation), covariances)
    return states, covariances


def _is_update_time(time):
    return time % noise_update_rate == 0


def _update_noises(state):
    """ Update the dynamic and measurement noises with respect to the current
    and previous states.
    """
    dynamic = state - dot(propag, transpose([_statehist[-1]]))
    dynamic = diag(dynamic.reshape(len(dynamic)))
    measure = dot(observation, state) - dot(observation, _statehist[-1])
    # dynamic += _dyn_noise
    # measure += _measurenoise
    return dynamic, measure


def _append_history(states, covariances):
    """ Save the prediction and covariances for the states at an instante """
    QTD_STATES = states.shape[0]
    _statehist.append(append([], states.T))
    _covariancehist.append([covariances[i, i] for i in range(QTD_STATES)])
