import numpy as np 


def solve(corr, order=None):
    order = corr.size - 1 if order is None else order
    errors = np.zeros(order + 1)
    alpha = np.ones(order + 1)
    # Base case
    corr = np.append(corr, 1)
    errors[0] = corr[0]
    alpha[0] = 1

    for i in range(1, order + 1):
        alpha_coef = corr[i]
        for j in range(1, i):
            alpha_coef += -alpha[j] * corr[i-j]

        ki = alpha_coef / errors[i-1]
        alpha[i] = ki

        for j in range(1, i):
            alpha[j] += -ki*alpha[i-j]
        errors[i] = (1-ki**2) * errors[i-1]
    return alpha, errors

