from math import cos, sin, pi
from sample.estimation import stdkalman
from sample.data import open_matrix, rsfile
from sample.plt_utils import DataSet, tsets
from sample.identification import Structure, Stage
from numpy import matrix, dot, arange, savetxt
import matplotlib.pyplot as plt
# import pdb

Px = 0
Py = 1


def estimate_simple_car():
    with open_matrix(tsets[DataSet.SIMPLE_CAR], ' ') as data:

        print('Initializing...')
        _initialise_simple_model()
        print('Computing Kalman...')
        estimated_positions = _filtrate_car(data[:, 1], matrix([[0], [0]]))
        savetxt('results/simple_car{}{}'.format(0, 0), estimated_positions.T)
        print('Finished!')


def estimate_bidirectional_car():
    print('Initialising...')
    _initialise_bidirectional_model()
    print('Filtrating...')
    print('Done!')


def _initialise_bidirectional_model():
    pass


def _initialise_simple_model():
    ANGLE = 32 * (pi / 180)
    DELTA_T = 1
    stdkalman.propag = matrix([[1, DELTA_T], [0, 1]])
    stdkalman.observation = matrix([[cos(ANGLE), 0], [sin(ANGLE), 0]])
    stdkalman.entry = matrix([[0, 0], [0, 0]])
    stdkalman.dyn_noise = matrix([[0.5 ** 2, 0], [0, 1.5 ** 2]])
    stdkalman.measurenoise = matrix([[5 ** 2, 0], [0, 5 ** 2]])


def _filtrate_car(measures, inputs):
    states, errors = stdkalman.filtrate(measures, inputs)
    states = matrix(states)
    return dot(stdkalman.observation, states.T)


def _save_dots(estimation, fname):

    with rsfile(fname, Stage.VALIDATING, Structure.SIMPLE_CAR, False) as rs:
        estimation.T.tofile(rs, '\t')


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
        # y_pos01 = r01[:, Py]
        x_pos10 = r10[:, Px]
        # y_pos10 = r10[:, Py]

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


estimate_simple_car()
# plot()
# plot_time()
