from math import cos, sin, pi
from sample.estimation import kalman_filter
from sample.data import open_matrix, rsfile
from sample.plt_utils import DataSet, tsets
from sample.identification import Structure, Stage
from numpy import matrix, dot, arange
import matplotlib.pyplot as plt

Px = 0
Py = 1


def estim():
    with open_matrix(tsets[DataSet.SIMPLE_CAR], ' ') as data, \
            rsfile('car', Stage.VALIDATING, Structure.SIMPLE_CAR) as results:

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
        states = kalman_filter(prop, entry, obs_matrix, u,
                               data[:, 1:], meas_noise, dyn_noise)
        states = matrix(states)
        estimated_positions = dot(obs_matrix, states.T)
        for position in estimated_positions.T:
            results.write('{} {}\n'.format(position[0, Px], position[0, Py]))
        print('Finished!')


def plot():
    with open_matrix('results/scar_car_val.rs', ' ') as results, \
            open_matrix('examples/car.dat', ' ') as measures:
        x_pos = results[:, 0]
        y_pos = results[:, 1]

        xmeasure = measures[:, 1]
        ymeasure = measures[:, 2]

        plt.figure()
        plt.plot(xmeasure, ymeasure, 'bo', markersize=2)
        plt.plot(x_pos, y_pos, 'r:', linewidth=1.5)
        plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
        plt.show()


def plot_xtime():
    with open_matrix('results/scar_car_val.rs', ' ') as results, \
            open_matrix('examples/car.dat', ' ') as measures:
        x_pos = results[:, 0]
        xmeasure = measures[:, 1]
        plt.figure()
        plt.plot(xmeasure, 'bo', markersize=4, fillstyle='none')
        plt.plot(x_pos, 'r-.', linewidth=1.5)
        plt.grid(True)
        plt.xticks(arange(0, 110, step=10))
        plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
        plt.show()


# estim()
# plot()
plot_xtime()
