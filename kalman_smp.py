from math import cos, sin, pi, atan
from sample.estimation.lse import clamp
from sample.estimation import kf, ekf
from sample.data import open_matrix, rsfile
from sample.plt_utils import DataSet, tsets
from sample.identification import Structure, Stage
from numpy import matrix, dot, savetxt, zeros, loadtxt, array
import matplotlib.pyplot as plt
# import pdb

Px = 0
Py = 1


def estimate_simple_car():
    with open_matrix(tsets[DataSet.SIMPLE_CAR], ' ') as data:

        print('Initializing...')
        _initialise_simple_model()
        print('Computing Kalman...')
        estimated_positions = _filtrate_kf_car(data[:, 1:], matrix([[0], [0]]))
        savetxt('results/simple_car{}{}'.format(0, 0), estimated_positions.T)
        print('Finished!')


def estimate_bidirectional_car():
    data = loadtxt(tsets[DataSet.BIDIRECTIONAL_CAR], usecols=(1, 2))
    print('Initialising...')
    _initialise_bidirectional_model()
    print('Filtrating...')
    estimated_positions = _filtrate_ekf_car(data, zeros((5, 1)))
    savetxt('results/bid_car_rs.txt', estimated_positions.T)
    print('Done!')


def _initialise_bidirectional_model():
    ANGLE = 50 * (pi / 180)
    DELTA_T = 0.1
    SPEED = __speed(DELTA_T, 0.7)
    # states = [X, Y, Theta, Vs, As]
    ekf.propag = array(
        [[1, 0, DELTA_T * cos(ANGLE), -DELTA_T * SPEED * sin(ANGLE), 0],  # X
         [0, 1, DELTA_T * sin(ANGLE), DELTA_T * SPEED * cos(ANGLE), 0],  # Y
         [0, 0, 1, 0, DELTA_T],  # Theta
         [0, 0, 0, SPEED, 0],  # Vetorial speed
         [0, 0, 0, 0, ANGLE]]  # Angular speed
    )
    ekf.observation = array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0]])
    ekf.entry = zeros((5, 5))
    ekf.noise_update_rate = 1


def _initialise_simple_model():
    ANGLE = 32 * (pi / 180)
    DELTA_T = 1
    kf.propag = matrix([[1, DELTA_T], [0, 1]])
    kf.observation = matrix([[cos(ANGLE), 0], [sin(ANGLE), 0]])
    kf.entry = matrix([[0, 0], [0, 0]])
    kf.dyn_noise = matrix([[0.5 ** 2, 0], [0, 1.5 ** 2]])
    kf.measurenoise = matrix([[5 ** 2, 0], [0, 5 ** 2]])


def _filtrate_kf_car(measures, inputs):
    states, errors = kf.filtrate(measures, inputs)
    states = matrix(states)
    return dot(kf.observation, states.T)


def _filtrate_ekf_car(measures, inputs):
    states, errors = ekf.filtrate(measures, inputs)
    states = matrix(states)
    observation = array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0]])
    return dot(observation, states.T)


def _save_dots(estimation, fname):

    with rsfile(fname, Stage.VALIDATING, Structure.SIMPLE_CAR, False) as rs:
        estimation.T.tofile(rs, '\t')


def plot_movement():
    measures = loadtxt(tsets[DataSet.BIDIRECTIONAL_CAR], usecols=(1, 2))
    estimation = loadtxt('results/bid_car_rs.txt')
    xestim = estimation[:, Px]
    yestim = estimation[:, Py]

    xmeasure = measures[:, Px]
    ymeasure = measures[:, Py]

    plt.figure()
    plt.plot(xmeasure, ymeasure, 'bo', markersize=2)
    plt.plot(xestim, yestim, 'r-', linewidth=1.5)
    plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
    plt.show()


def plot_time():
    measures = loadtxt(tsets[DataSet.BIDIRECTIONAL_CAR], usecols=(1, 2))
    estimation = loadtxt('results/bid_car_rs.txt')
    xestim = estimation[:, Px]
    yestim = estimation[:, Py]
    xmeasure = measures[:, Px]
    ymeasure = measures[:, Py]

    plt.figure()
    plt.subplot(211)
    plt.ylabel('Posição em X')
    plt.plot(xmeasure, 'c-.', linewidth=1)
    plt.plot(xestim, 'r-', linewidth=1)

    plt.subplot(212)
    plt.plot(ymeasure, 'c-.', linewidth=1)
    plt.plot(yestim, 'r-', linewidth=1)
    plt.ylabel('Posição em Y')
    plt.xlabel('Tempo')
    plt.subplots_adjust(0.08, 0.125, 0.98, 0.95, None, 0.3)
    plt.show()


def plot_angle():
    estimation = loadtxt('results/bid_car_rs.txt')
    theta = estimation[:, 2]

    plt.plot(theta)
    plt.show()


def __speed(time_in_sec, fraction=1):
    """ Compute the maximum velocity based on Porsche 911 (997) Turbo S

    Args:
        time_in_sec (float): The time variation
        fraction (float): How fast the car is compared to the Porsche
    """
    fraction = clamp(fraction, min_=0, max_=1.0)
    return 100 * time_in_sec / 2.7 * fraction


# estimate_simple_car()
# estimate_bidirectional_car()
# plot_time()
# plot_movement()
plot_angle()
