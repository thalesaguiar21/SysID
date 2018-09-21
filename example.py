import data.utils as dut
import sample.identification as sid
import sample.metrics as met
from numpy import append, matrix, dot
import matplotlib.pyplot as plt
# import pdb

tsets = {'bbeam': 'training/ballbeam_train.dat',
         'dryer': 'training/dryer_train.dat',
         'tank1': 'training/tank1_train.dat'}

valsets = {'bbeam': 'validation/ballbeam_val.dat',
           'dryer': 'validation/dryer_val.dat',
           'tank1': 'validation/tank1_val.dat'}


def train(fname, order, delay, inp=0, out=1, est='arx'):
    f_templ = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    rs_file = 'results/' + fname + '_train_' + est
    with open(rs_file + '.rs', 'w') as file:
        header = ['stdev', 'aic', 'fpe', 'order', 'delay', 'params']
        file.write(f_templ.format(*header))
        for order in xrange(1, order + 1):
            for delay in xrange(0, delay + 1):
                theta = []
                res = []
                u, y = dut.r_dots(tsets[fname], inp, out, '\t')
                if est == 'arx':
                    theta, res, _ = sid.identify_arx(u, y, order, delay)
                else:
                    theta, res, _ = sid.identify_armax(u, y, order, delay)
                stdev = format(met.stdev(res), '.3e')
                aic = format(met.aic(res, theta.size), '.3e')
                fpe = format(met.fpe(res, theta.size), '.3e')
                t = [theta[i, 0] for i in range(theta.shape[0])]
                file.write(f_templ.format(stdev, aic, fpe, order, delay, t))


def validate(fname, order, delay, theta, scatrate=3, inp=0, out=1, est='arx'):
    u, y = dut.r_dots(valsets[fname], inp, out, '\t')
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
    plt.plot(range(0, estim.size - 1, scatrate), scatreal, 'rx', markersize=4)
    plt.plot(estim, 'k--', linewidth=1)

    plt.subplots_adjust(0.08, 0.2, 0.98, 0.95, None, 0.3)
    plt.show()


def __gen_history(fname, order, delay, inp, out, est='arx'):
    u, y = dut.r_dots(fname, inp, out, '\t')
    phist = []
    if est == 'arx':
        _, _, _, phist = sid.identify_arx_rec(u, y, order, delay)
    elif est == 'armax':
        _, _, _, phist = sid.identify_armax_rec(u, y, order, delay)
    else:
        raise ValueError('Could no identify structure: ' + est)
    phist = matrix(phist)
    hist = [append([], phist[:, i]) for i in range(phist.shape[1])]
    return hist


def plot_hist(fname, order, delay, est='arx', inp=0, out=1, smp=None):
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    tarx = __gen_history(tsets[fname], order, delay, inp, out, est='arx')
    tarmax = __gen_history(tsets[fname], order, delay, inp, out, est='armax')
    smp = len(tarx[0]) if smp is None else smp

    for i in xrange(len(tarx)):
        print tarx[i][-1]

    plt.figure(1, figsize=(8, 6))
    plt.subplot(211)
    plt.title('ARX')
    plt.ylabel('Valor do parametro')

    for i in xrange(2 * order):
        lab = 'p' + str(i + 1)
        plt.plot(tarx[i][:smp], linewidth=1.5, color=colors[i], label=lab)
        plt.legend()

    # Configura o subplot para armax
    plt.subplot(212)
    plt.title('ARMAX')
    plt.ylabel('Valor do parametro')
    plt.xlabel('Tempo')

    for i in xrange(3 * order):
        lab = 'p' + str(i + 1)
        plt.plot(tarmax[i][:smp], linewidth=1.5, color=colors[i], label=lab)
        plt.legend()

    plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
    plt.show()


theta = matrix([1.0000000000000009, -5.4778230562657626e-15]).T

# train('tank1', 3, 8, inp=0, out=2)
# train('tank1', 3, 3, inp=0, out=2, est='armax')
validate('tank1', 1, 3, theta, inp=0, out=2, scatrate=2)
# plot_hist('tank1', 1, 3, inp=0, out=2, smp=100)
