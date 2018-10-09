import matplotlib.pyplot as plt
import sample.identification as sid
import sample.estimation as solv
from numpy import matrix, append
from sample.data import open_matrix
from enum import Enum

""" This module is composed of some plotting methods to create consistent
plots along the produced work
"""


class DataSet(Enum):
    """ Available data sets """
    BBEAM = 1
    DRYER = 2
    TANK = 3
    IPCA_A = 4
    IPCA_B = 5
    IPCA_C = 6


__left = 0.08
__right = 0.98
__bot = 0.125
__top = 0.95
__ws = None
__hs = 0.3

tsets = {DataSet.BBEAM: 'examples/training/ballbeam_train.dat',
         DataSet.DRYER: 'examples/training/dryer_train.dat',
         DataSet.TANK: 'examples/training/tank1_train.dat',
         DataSet.IPCA_A: 'examples/ipca1.dat',
         DataSet.IPCA_B: 'examples/ipca2.dat',
         DataSet.IPCA_C: 'examples/ipca3.dat'}

vsets = {DataSet.BBEAM: 'examples/validation/ballbeam_val.dat',
         DataSet.DRYER: 'examples/validation/dryer_val.dat',
         DataSet.TANK: 'examples/validation/tank1_val.dat',
         DataSet.IPCA_A: 'examples/ipca1.dat',
         DataSet.IPCA_B: 'examples/ipca2.dat',
         DataSet.IPCA_C: 'examples/ipca3.dat'}


def gen_history(fname, order, delay, inp, est=sid.Structure.ARX):
    with open_matrix(vsets[fname]) as dat:
        phist = []
        if est == sid.Structure.ARX:
            A, B = sid.idarx(dat, order, delay)
            _, phist = solv.recursive_lse(A, B)
        elif est == sid.Structure.ARMAX:
            _, _, _, phist = sid.idarmaxr(dat, order, delay)
        else:
            raise ValueError('Unknown structure: ' + est)
        phist = matrix(phist)
        hist = [append([], phist[:, i]) for i in range(phist.shape[1])]
        return hist


def plot_history(tarx, tarmax, smp=None):
    smp = len(tarx[0]) if smp is None else smp

    plt.figure(1, figsize=(8, 6))
    plt.subplot(211)
    plt.title('ARX')
    plt.ylabel('Valor do parametro')

    for i in range(len(tarx)):
        lab = 'p' + str(i + 1)
        plt.plot(tarx[i][:smp], linewidth=1.5, label=lab)
        plt.legend()

    # Configura o subplot para armax
    plt.subplot(212)
    plt.title('ARMAX')
    plt.ylabel('Valor do parametro')
    plt.xlabel('Tempo')

    for i in range(len(tarmax)):
        lab = 'p' + str(i + 1)
        plt.plot(tarmax[i][:smp], linewidth=1.5, label=lab)
        plt.legend()

    plt.subplots_adjust(__left, __bot, __right, __top, __ws, __hs)
    plt.show()
