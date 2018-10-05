from math import sqrt, log
from numpy import dot


def stdev(samples):
    """ Compute the standard deviation of the given samples """
    mean = sum(samples) / samples.size
    variances = [(s - mean) ** 2 for s in samples]
    return sqrt(sum(variances) / len(samples))


def squared(samples):
    """ Compute the sum of the modulo of the samples """
    sqr_samples = dot(samples.T, samples)
    total = 0
    for samp in sqr_samples:
        total += sqrt(samp)
    return total


def aic(samples, nt):
    """ Compute the Akaike criterion of the samples """
    return samples.size * log(stdev(samples) ** 2) + 2 * nt


def fpe(samples, nt):
    """ Compute the Final Prediction Error (FPE) of the samples """
    ndots = samples.size
    sterm = ndots * log((ndots + nt) / (ndots - nt))
    return ndots * log(stdev(samples) ** 2) + sterm


def bic(samples, nt):
    """ Compute the Bayesian Information Criterion of the samples """
    ndots = samples.size
    sterm = nt * log(ndots)
    return ndots * log(stdev(samples) ** 2) + sterm
