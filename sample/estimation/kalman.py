from numpy import zeros, dot, eye, append, transpose, diag
from numpy.linalg import inv


class Linear(object):

    def __init__(self, propag, entry, observation, initialcovariance=1000):
        self.propag = propag
        self.entry = entry
        self.observation = observation
        self._QTD_STATES = propag.shape[0]
        _QTD_RESULTS = observation.shape[0]
        self._states = zeros((self._QTD_STATES, 1))
        self.covariances = eye(self._QTD_STATES) * initialcovariance
        self.dyn_noise = zeros((self._QTD_STATES, self._QTD_STATES))
        self.measurenoise = zeros((_QTD_RESULTS, _QTD_RESULTS))
        self._statehist = []
        self._covariancehist = []

    def filtrate(self, measures, inputs):
        for t in range(len(measures)):
            self._propagate(inputs)
            self._adjust(measures[t])
            self._append_history()
        return self._statehist, self._covariancehist

    def _propagate(self, inputs):
        """ Propagate the states and covariances to the next instant """
        self._states = dot(self.propag, self._states) + dot(self.entry, inputs)
        self.covariances = dot(
            self.propag, dot(self.covariances, self.propag.T))
        self.covariances += self.dyn_noise

    def _adjust(self, measures):
        """ Adjust the predictions with the measurementes """
        term = dot(self.covariances, self.observation.T)
        gain = dot(term, inv(dot(self.observation, term) + self.measurenoise))
        self._states += dot(gain, transpose([measures]) -
                            dot(self.observation, self.states))
        self.covariances -= dot(dot(gain, self.observation), self.covariances)

    def _append_history(self):
        """ Save the prediction and covariances at an instant """
        self._statehist.append(append([], self.states.T))
        self._covariancehist.append(
            [self.covariances[i, i] for i in range(self._QTD_STATES)]
        )


class Extended(Linear):
    pass
