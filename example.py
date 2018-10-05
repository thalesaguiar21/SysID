from sample.data import rsfile, open_matrix
from sample import estimation as solv
from sample import identification as sid
from sample import metrics as met
from numpy import append, matrix, dot
import matplotlib.pyplot as plt
import pdb

tsets = {'bbeam': 'examples/training/ballbeam_train.dat',
         'dryer': 'examples/training/dryer_train.dat',
         'tank1': 'examples/training/tank1_train.dat',
         'ipca1': 'examples/ipca1.dat',
         'ipca2': 'examples/ipca2.dat',
         'ipca3': 'examples/ipca3.dat'}

valsets = {'bbeam': 'examples/validation/ballbeam_val.dat',
           'dryer': 'examples/validation/dryer_val.dat',
           'tank1': 'examples/validation/tank1_val.dat',
           'ipca1': 'examples/ipca1.dat',
           'ipca2': 'examples/ipca2.dat',
           'ipca3': 'examples/ipca3.dat'}


def __comp_residue(theta, A, B):
    ypred = dot(A, theta)
    return ypred, B - ypred


def train_rec(fname, order, atr, conf=1000, ffac=1.0, est='arx'):
    with rsfile(fname, 'rtr', est) as rs, open_matrix(tsets[fname]) as dt:
        for order in range(1, order + 1):
            for atr in range(0, atr + 1):
                theta = []
                res = []
                if est == 'arx':
                    A, B = sid.idarx(dt, order, atr)
                    theta, _ = solv.recursive_lse(A, B, conf, ffac)
                    _, res = __comp_residue(theta, A, B)
                else:
                    theta, res, _, _ = sid.idarmaxr(dt, order, atr, conf, ffac)
                __save_results(rs, res, theta, order, atr)


def train(fname, order, delay, est='arx'):
    with rsfile(fname, sys=est) as rs, open_matrix(tsets[fname]) as dt:
        for order in range(1, order + 1):
            for delay in range(0, delay + 1):
                theta = []
                res = []
                if est == 'arx':
                    A, B = sid.idarx(dt, order, delay)
                    theta, res, _ = solv.mat_lse(A, B)
                else:
                    theta, res, _ = sid.idarmax(dt, order, delay)
                __save_results(rs, res, theta, order, delay)


def __save_results(rfile, res, theta, order, delay):
    fileline = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    stdev = format(met.stdev(res), '.3e')
    aic = format(met.aic(res, theta.size), '.3e')
    fpe = format(met.fpe(res, theta.size), '.3e')
    t = [theta[i, 0] for i in range(theta.shape[0])]
    rfile.write(fileline.format(stdev, aic, fpe, order, delay, t))


def validate(fname, order, delay, theta, scatrate=3, inp=[0, 1], est='arx'):
    with open_matrix(valsets[fname]) as dat:
        reg, B = sid.idarx(dat, order, delay)
        ypred = dot(reg, theta)
        print('{:.3e}'.format(met.stdev(B - ypred)))
        print('{:.3e}'.format(met.aic(B - ypred, 2 * order)))
        __plot_test(append([], dat[:, -1:]), append([], ypred), scatrate)


def __plot_test(real, estim, scatrate):
    estpts = real.size - estim.size
    plt.figure(figsize=(8, 3))
    plt.xlabel('Amostras')
    real = real[estpts:]
    scatreal = [real[i * scatrate] for i in range((estim.size) / scatrate)]
    plt.plot(range(0, estim.size - 1, scatrate), scatreal, 'ro', markersize=2)
    plt.plot(estim, 'k:', linewidth=1)

    plt.subplots_adjust(0.08, 0.2, 0.98, 0.95, None, 0.3)
    plt.show()


def __gen_history(fname, order, delay, inp, est='arx'):
    with open_matrix(valsets[fname]) as dat:
        phist = []
        if est == 'arx':
            A, B = sid.idarx(dat, order, delay)
            _, phist = solv.recursive_lse(A, B)
        elif est == 'armax':
            _, _, _, phist = sid.idarmaxr(dat, order, delay)
        else:
            raise ValueError('Could no identify structure: ' + est)
        phist = matrix(phist)
        hist = [append([], phist[:, i]) for i in range(phist.shape[1])]
        return hist


def plot_hist(fname, order, delay, est='arx', inp=[0, 1], smp=None):
    # colors = ['b', 'g', 'r', 'm', 'y', 'k']
    tarx = __gen_history(tsets[fname], order, delay, inp, est='arx')
    tarmax = __gen_history(tsets[fname], order, delay, inp, est='armax')
    smp = len(tarx[0]) if smp is None else smp

    for i in range(len(tarx)):
        print(tarx[i][-1])

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

    plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
    plt.show()


theta = matrix([0.9999999998278026, 1.2894267218843419e-10]).T

# train_rec('ipca1', 5, 0, inp=[0, 1], ffac=.95)
# train_rec('ipca2', 5, 0, inp=[0, 1, 2], ffac=.95)
# train_rec('ipca3', 5, 0, inp=[0, 1, 2, 3], ffac=.95)
# train_rec('ipca1', 5, 0, inp=[0, 1], est='armax', ffac=.95)
# train_rec('ipca2', 5, 0, inp=[0, 1, 2], est='armax', ffac=.95)
# train_rec('ipca3', 5, 0, inp=[0, 1, 2, 3], est='armax', ffac=.95)
# train_rec('bbeam', 3, 3)
# train('bbeam', 3, 3)
# train('tank1', 3, 3, inp=0, out=2, est='armax')
validate('bbeam', 1, 0, theta, inp=[0, 1], scatrate=2)
# plot_hist('ipca1', 1, 0, inp=[0, 1])
