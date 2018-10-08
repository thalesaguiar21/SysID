import matplotlib.pyplot as plt
import sample.identification as sid
import sample.estimation as solv
from numpy import matrix, append
from sample.data import open_matrix

""" This module is composed of some plotting methods to create consistent
plots along the produced work
"""


class SetAlias(object):
    """ Available data sets """
    pass


BBEAM = SetAlias()
DRYER = SetAlias()
TANK = SetAlias()
IPCA_A = SetAlias()
IPCA_B = SetAlias()
IPCA_C = SetAlias()
CAR_A = SetAlias()


__left = 0.08
__right = 0.98
__bot = 0.125
__top = 0.95
__ws = None
__hs = 0.3

tsets = {BBEAM: 'examples/training/ballbeam_train.dat',
         DRYER: 'examples/training/dryer_train.dat',
         TANK: 'examples/training/tank1_train.dat',
         IPCA_A: 'examples/ipca1.dat',
         IPCA_B: 'examples/ipca2.dat',
         IPCA_C: 'examples/ipca3.dat',
         CAR_A: 'examples/car.dat'}

vsets = {BBEAM: 'examples/validation/ballbeam_val.dat',
         DRYER: 'examples/validation/dryer_val.dat',
         TANK: 'examples/validation/tank1_val.dat',
         IPCA_A: 'examples/ipca1.dat',
         IPCA_B: 'examples/ipca2.dat',
         IPCA_C: 'examples/ipca3.dat',
         CAR_A: 'examples/car.dat'}


def gen_history(fname, order, delay, inp, est=sid.ARX):
    with open_matrix(vsets[fname]) as dat:
        phist = []
        if est == sid.ARX:
            A, B = sid.idarx(dat, order, delay)
            _, phist = solv.recursive_lse(A, B)
        elif est == sid.ARMAX:
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
