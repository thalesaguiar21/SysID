import matplotlib.pyplot as plt
import sample.particle.sampling as sampl
import sample.particle.structures as st
import sample.particle.initers as inits
import sample.particle.interface as ui
from math import sin, cos
from numpy.random import normal
from sample.utils import clip


MIN_PARTICLES = 1
MAX_PARTICLES = 10000
NROBOTS = 4


class RobotMeasures:

    def __init__(self, linvels, angvels, measures):
        self.linvels = linvels
        self.angvels = angvels
        self.measures = measures


def filtrateandplot(data, nparticles, gap, init, smp):
    rbtdata = _measures_from_data(data)
    nparticles = clip(nparticles, MIN_PARTICLES, MAX_PARTICLES)
    particles = inits.initialise(nparticles, NROBOTS, data[0], init)
    total = len(data) - 1
    fig, ax = plt.subplots()
    for i in range(1, len(data)):
        ui.redraw(i, total, particles, ax)
        st.evalparticles(particles, rbtdata.measures[i])
        particles = sampl.resample(particles, nparticles, gap, smp)
        _propagate(particles, rbtdata.linvels[i], rbtdata.angvels[i])


def _propagate(particles, lin_vels, ang_vels, dt=1.):
    for part in particles:
        for rbt, vlin, vang in zip(part.robots, lin_vels, ang_vels):
            vlin_ = vlin + normal(0, 0.09 * abs(vlin) + 0.01 * abs(vang))
            vang_ = vang + normal(0, 0.01 * abs(vlin) + 0.09 * abs(vang))
            gamma = normal(0, 0.005 * abs(vlin) + 0.05 * abs(vang))
            prop = vlin_ / vang_
            rbt.x -= prop * (sin(rbt.ang) - sin(rbt.ang + vang_ * dt))
            rbt.y += prop * (cos(rbt.ang) - cos(rbt.ang + vang_ * dt))
            rbt.ang += vang_ * dt + gamma * dt
            st.put_in_limits(rbt)


def _measures_from_data(data):
    linvels = _extract_from_data(data, 8, 10)
    angvels = _extract_from_data(data, 9, 10)
    measures = _extract_robots_measure(data)
    return RobotMeasures(linvels, angvels, measures)


def _extract_robots_measure(data):
    rbt_meas = []
    for measure in data:
        rbt_meas.append([measure[10 * i:8 + i * 10] for i in range(NROBOTS)])
    return rbt_meas


def _extract_from_data(data, pos, step):
    info = []
    for measure in data:
        info.append([measure[pos + i * step] for i in range(NROBOTS)])
    return info
