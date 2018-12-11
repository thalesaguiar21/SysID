import matplotlib.pyplot as plt
from math import floor

XCOL = 0
YCOL = 1
BARLENGTH = 20


def redraw(i, total, particles, ax):
    _progress(i, total, particles)
    _particlechart(particles, ax)


def _progress(i, total, particles):
    percent = i / total
    complete = floor(percent * BARLENGTH)
    remaining = BARLENGTH - complete
    bar = '{}{}'.format(complete * '#', remaining * '-')
    message = 'Filtrating | {} | {:.2f} %'.format(bar, percent * 100)
    message += '\tNR: ' + str(_total_robots(particles))
    message += '\tNP: ' + str(len(particles))
    print(message, end='\r')


def _total_robots(particles):
    k = 0
    for p in particles:
        k += len(p.robots)
    return k


def _particlechart(particles, ax):
    xs, ys = _robots_coords(particles)
    ax.clear()
    ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], 'k-')
    ax.plot([0, 0, 1, -1], [1, -1, 0, 0], 'ks')
    ax.plot(xs, ys, 'go', markersize=1)
    plt.pause(0.0001)


def _robots_coords(particles):
    xs = []
    ys = []
    for p in particles:
        xs.extend(p.robots_xs())
        ys.extend(p.robots_ys())
    return xs, ys
