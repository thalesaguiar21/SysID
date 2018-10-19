from numpy import dot


def clamp(value, min_, max_):
    return max(min(value, max_), min_)


def calc_residues(psi, y, theta):
    """ Calculate the residues from the predicted theta """
    ypredicted = dot(psi, theta)
    return y - ypredicted
