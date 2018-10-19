from numpy import zeros, dot, eye, append
from numpy.linalg import inv
# import pdb

initital_covariance = 1000
statehist = []
covariancehist = []
propag = None
entry = None
observation = None
dyn_noise = None
measurenoise = None


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
        __append_history(states, covariances)
    return statehist, covariancehist


def __propagate(states, covariances, inputs):
    """ Propagate the states and covariances to the next instant in time """
    states = dot(propag, states) + dot(entry, inputs)
    covariances = dot(propag, dot(covariances, propag.T)) + dyn_noise
    return states, covariances


def __adjust(states, covariances, measures):
    """ Adjust the predictions with the measurementes """
    term = dot(covariances, observation.T)
    gain = dot(term, inv(dot(observation, term) + measurenoise))
    states = states + dot(gain, measures.T - dot(observation, states))
    covariances = covariances - dot(dot(gain, observation), covariances)
    return states, covariances


def __append_history(states, covariances):
    """ Save the prediction and covariances for the states at an instante """
    QTD_STATES = states.shape[0]
    statehist.append(append([], states.T))
    covariancehist.append([covariances[i, i] for i in range(QTD_STATES)])
