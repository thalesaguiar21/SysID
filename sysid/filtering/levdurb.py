import numpy as np 


def solve(margs):
    order = margs.size
    R = np.append(margs, 0)
    errors = np.zeros(order + 1) + 1e-15
    errors[0] = margs[0]

    icogs = np.zeros(order + 1)
    icogs[0] = 1
    for i in range(1, order+1):
        alpha_R = 0
        for j in range(1, i-1):
            alpha_R += icogs[j] * R[i-j]
        ki = (R[i]-alpha_R) / errors[i-1]
        icogs[i] = ki
        if i > 1:
            icogs[1:i] -= ki * np.flip(icogs[1:i])
        errors[i] = (1 - ki**2)*errors[i]
    return icogs


def diag(mtx):
    mdiag = np.zeros(mtx.shape)
    for i in range(mtx.shape[0]):
        mdiag[i, i] = mtx[i, i]
    return mdiag

