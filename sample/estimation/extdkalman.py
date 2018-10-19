from numpy import zeros, dot, eye, append
from numpy.linalg import inv
# import pdb

initital_covariance = 1000
propag = None
entry = None
observation = None
noise_update_rate = 10
_statehist = []
_covariancehist = []
_dyn_noise = None
_measurenoise = None


def filtrate(measures, inputs):
    """ Estimate the states and covariances for the kalman model. The propagati
    on, entry and observation matrices must be given before calling this method

    Args:
        measures (ndarray): The sensor measurements
        inputs (ndarray): The exogenous inputs

    Returns:
        The states and covariances through time
    """
    states = zeros((propag.shape[0], 1))
    covariances = eye(propag.shape[0]) * initital_covariance
    for t in range(len(measures)):
        states, covariances = __propagate(states, covariances, inputs)
        states, covariances = __adjust(states, covariances, measures[t])
        if is_update_time(t):
            _dyn_noise, _measurenoise = __update_noises(states)
        __append_history(states, covariances)
    return _statehist, _covariancehist


def __propagate(states, covariances, inputs):
    """ Propagate the states and covariances to the next instant in time """
    states = dot(propag, states) + dot(entry, inputs)
    covariances = dot(propag, dot(covariances, propag.T)) + _dyn_noise
    return states, covariances


def __adjust(states, covariances, measures):
    """ Adjust the predictions with the measurementes """
    term = dot(covariances, observation.T)
    gain = dot(term, inv(dot(observation, term) + _measurenoise))
    states = states + dot(gain, measures.T - dot(observation, states))
    covariances = covariances - dot(dot(gain, observation), covariances)
    return states, covariances


def is_update_time(time, update_rate):
    return time % update_rate == 0


def __update_noises(state):
    """ Update the dynamic and measurement noises with respect to the current
    and previous states.
    """
    estimated_states = dot(observation, state)
    dynamic = _statehist[-1] - estimated_states
    measure = dot(observation, _statehist[-1]) - estimated_states
    return dynamic, measure


def __append_history(states, covariances):
    """ Save the prediction and covariances for the states at an instante """
    QTD_STATES = states.shape[0]
    _statehist.append(append([], states.T))
    _covariancehist.append([covariances[i, i] for i in range(QTD_STATES)])
