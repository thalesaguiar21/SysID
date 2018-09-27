import data.utils as dut
import sample.identification as sid
import sample.metrics as met
from numpy import append, matrix, dot
import matplotlib.pyplot as plt
import pdb

tsets = {'bbeam': 'training/ballbeam_train.dat',
         'dryer': 'training/dryer_train.dat',
         'tank1': 'training/tank1_train.dat',
         'ipca1': 'ipca1.dat',
         'ipca2': 'ipca2.dat',
         'ipca3': 'ipca3.dat'}

valsets = {'bbeam': 'validation/ballbeam_val.dat',
           'dryer': 'validation/dryer_val.dat',
           'tank1': 'validation/tank1_val.dat',
           'ipca1': 'ipca1.dat',
           'ipca2': 'ipca2.dat',
           'ipca3': 'ipca3.dat'}


def train_rec(fname, order, atr, inp=[0, 1], ffac=1.0, est='arx'):
    f_templ = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    rs_file = 'results/' + fname + '_train_' + est
    with open(rs_file + '.rs', 'w') as file:
        header = ['stdev', 'aic', 'fpe', 'order', 'atr', 'params']
        file.write(f_templ.format(*header))
        for order in range(1, order + 1):
            for atr in range(0, atr + 1):
                theta = []
                res = []
                u, y = dut.r_dots(tsets[fname], inp, '\t')
                if est == 'arx':
                    theta, res, _, _ = sid.identify_arx_rec(
                        u, y, order, atr)
                else:
                    theta, res, _, _ = sid.identify_armax_rec(
                        u, y, order, atr)
                stdev = format(met.stdev(res), '.3e')
                aic = format(met.aic(res, theta.size), '.3e')
                fpe = format(met.fpe(res, theta.size), '.3e')
                t = [theta[i, 0] for i in range(theta.shape[0])]
                file.write(f_templ.format(stdev, aic, fpe, order, atr, t))


def train(fname, order, delay, inp=[0, 1], est='arx'):
    f_templ = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    rs_file = 'results/' + fname + '_train_' + est
    with open(rs_file + '.rs', 'w') as file:
        header = ['stdev', 'aic', 'fpe', 'order', 'delay', 'params']
        file.write(f_templ.format(*header))
        for order in range(1, order + 1):
            for delay in range(0, delay + 1):
                theta = []
                res = []
                u, y = dut.r_dots(tsets[fname], inp, '\t')
                if est == 'arx':
                    theta, res, _ = sid.identify_arx(u, y, order, delay)
                else:
                    theta, res, _ = sid.identify_armax(u, y, order, delay)
                stdev = format(met.stdev(res), '.3e')
                aic = format(met.aic(res, theta.size), '.3e')
                fpe = format(met.fpe(res, theta.size), '.3e')
                t = [theta[i, 0] for i in range(theta.shape[0])]
                file.write(f_templ.format(stdev, aic, fpe, order, delay, t))


def validate(fname, order, delay, theta, scatrate=3, inp=[0, 1], est='arx'):
    u, y = dut.r_dots(valsets[fname], inp, '\t')
    reg, B = sid.identify_arx_params(u, y, order, delay)
    ypred = dot(reg, theta)
    print '{:.3e}'.format(met.stdev(B - ypred))
    print '{:.3e}'.format(met.aic(B - ypred, 2 * order))
    __plot_test(append([], y), append([], ypred), scatrate)


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
    u, y = dut.r_dots(fname, inp, '\t')
    phist = []
    if est == 'arx':
        _, _, _, phist = sid.identify_arx_rec(
            u, y, order, delay)
    elif est == 'armax':
        _, _, _, phist = sid.identify_armax_rec(
            u, y, order, delay)
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
        print tarx[i][-1]

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
# train('tank1', 3, 8, inp=0, out=2)
# train('tank1', 3, 3, inp=0, out=2, est='armax')
# validate('ipca1', 1, 0, theta, inp=[0, 1], scatrate=2)
plot_hist('ipca1', 1, 0, inp=[0, 1])
