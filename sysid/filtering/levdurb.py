import numpy as np 


def solve(margs, indep):
    diagonal = diag(margs)


def diag(mtx):
    mdiag = np.zeros(mtx.shape)
    for i in range(mtx.shape[0]):
        mdiag[i, i] = mtx[i, i]
    return mdiag

