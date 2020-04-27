import numpy as np 


def solve(corr, order=None):
    order = corr.size if order is None else order
    errors = np.zeros(order)
    errors[0] = corr[0]
    alpha = np.zeros(order)
    alpha[0] = 1

    for i in range(1, order):
        alpha_coef = 0
        for j in range(1, i):
            alpha_coef += alpha[j] * corr[i-j]

        ki = (corr[i]-alpha_coef) / errors[i-1]
        alpha[i] = ki

        if i > 1:
            for j in range(1, i):
                alpha[j] -= ki*alpha[i-j]

        errors[i] = (1 - ki**2)*errors[i-1]
    return alpha, errors

