from math import cos, sin, pi
from sample.estimation import discrete_kalman
from sample.data import open_matrix, rsfile
from sample.plt_utils import DataSet, tsets
from sample.metrics import aic, stdev
from sample.identification import Structure, Stage
from numpy import matrix, dot, arange, append
import matplotlib.pyplot as plt
import pdb

Px = 0
Py = 1


def estim():
    with open_matrix(tsets[DataSet.SIMPLE_CAR], ' ') as data:

        print('Initializing...')
        ANGLE = 32 * (pi / 180)
        DELTA_T = 1
        prop = matrix([[1, DELTA_T], [0, 1]])
        entry = matrix([[0, 0], [0, 0]])
        u = matrix([[0], [0]])
        obs_matrix = matrix([[cos(ANGLE), 0], [sin(ANGLE), 0]])
        dyn_noise = matrix([[0.5 ** 2, 0], [0, 1.5 ** 2]])
        meas_noise = matrix([[5 ** 2, 0], [0, 5 ** 2]])

        print('Computing Kalman...')
        states, errors = discrete_kalman(prop, entry, obs_matrix, u,
                                         data[:, 1:], meas_noise, dyn_noise)
        pdb.set_trace()
        states = matrix(states)
        estimated_positions = dot(obs_matrix, states.T)
        _save_dots(estimated_positions, errors, 'car/car{}{}'.format(0, 0))
        print('Finished!')


def _save_dots(estimation, errors, fname):
    with rsfile(fname, Stage.VALIDATING, Structure.SIMPLE_CAR, False) as rs:
        for position, error in zip(estimation.T, errors):
            rs.write('{}\t{}'.format(position[0, Px], position[0, Py]))
            rs.write('\t{}\t{}\n'.format(*error))


def plot():
    with open_matrix('results/car/scar_car00_val.rs', ' ') as r00, \
            open_matrix('examples/car.dat', ' ') as measures:
        x_pos00 = r00[:, Px]
        y_pos00 = r00[:, Py]

        xmeasure = measures[:, 1]
        ymeasure = measures[:, 2]

        plt.figure()
        plt.plot(xmeasure, ymeasure, 'bo', markersize=2)
        plt.plot(x_pos00, y_pos00, 'r:', linewidth=1.5)
        plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
        plt.show()


def plot_time():
    with open_matrix('results/car/scar_car00_val.rs', ' ') as r00, \
            open_matrix('results/car/scar_car01_val.rs', ' ') as r01, \
            open_matrix('results/car/scar_car10_val.rs', ' ') as r10, \
            open_matrix('examples/car.dat', ' ') as measures:
        x_pos00 = r00[:, Px]
        y_pos00 = r00[:, Py]
        x_pos01 = r01[:, Px]
        y_pos01 = r01[:, Py]
        x_pos10 = r10[:, Px]
        y_pos10 = r10[:, Py]

        xmeasure = measures[:, 1]
        ymeasure = measures[:, 2]

        plt.figure()
        plt.subplot(211)
        plt.plot(xmeasure, 'bo', markersize=4, fillstyle='none')
        plt.plot(x_pos00, 'r-.', linewidth=1)
        plt.plot(x_pos01, 'm-.', linewidth=1)
        plt.plot(x_pos10, 'k-.', linewidth=1)
        plt.grid(True)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posição em x')
        plt.xticks(arange(0, 110, step=10))

        plt.subplot(212)
        plt.plot(ymeasure, 'go', markersize=4, fillstyle='none')
        plt.plot(y_pos00, 'C1-.', linewidth=1.5)
        plt.grid(True)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posição em y')
        plt.xticks(arange(0, 110, step=10))

        plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
        plt.show()


estim()
# plot()
# plot_time()
