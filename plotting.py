from sample.data import open_matrix
import matplotlib.pyplot as plt
# import pdb


def separate_xy():
    with open_matrix('examples/car.dat', ' ') as data:
        return data[:, 1], data[:, 2]


def plot_data(x, y):
    plt.plot(x, y, 'r-')
    plt.subplots_adjust(0.08, 0.2, 0.98, 0.95, None, 0.3)
    plt.show()


x, y = separate_xy()
plot_data(x, y)
