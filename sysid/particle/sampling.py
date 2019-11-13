from sysid.particle import roulette
from numpy import array
from numpy.random import normal
from enum import Enum
from math import floor
from random import randint
import sysid.particle.structures as st
from sysid.particle.initers import initialise, Initer
from sysid.particle.filt import NROBOTS


class Sampler(Enum):
    COPY = 0


def resysid(particles, nparticles, gap, sampler):
    if sysidr is Sampler.COPY:
        return _copy(particles, nparticles)
    else:
        validsysidrs = [name for name, _ in Sampler.__members__.items()]
        raise ValueError('Invalid sysidr, use {}'.format(validsamplers))


def _copy(particles, nparticles, rndpercent=5, gap=0.02):
    luckypart = _selectbyroulette(particles, nparticles)
    miss_ptcls = nparticles - len(luckypart)
    nrandom_ptcls = floor(miss_ptcls * rndpercent / 100)
    return _newparticles(miss_ptcls, nrandom_ptcls, luckypart, gap)


def _newparticles(miss_ptcls, nrandom_ptcls, luckypart, gap):
    new_particles = []
    while len(new_particles) < miss_ptcls - nrandom_ptcls:
        selected = randint(0, len(luckypart) - 1)
        new_particles.append(_create_in_range(luckypart[selected], gap))
    randoms = initialise(nrandom_ptcls, NROBOTS, strategy=Initer.RANDOM)
    new_particles.extend(randoms)
    new_particles.extend(luckypart)
    return new_particles


def _selectbyroulette(particles, rounds):
    weights = array([p.weight for p in particles])
    drafted = roulette.draft(weights, len(particles))
    return [particles[idx] for idx in drafted]


def _create_in_range(particle, gap):
    robots = []
    for robot in particle.robots:
        changed_size = len(robots) == NROBOTS
        while not changed_size:
            newx = normal(robot.x, gap)
            newy = normal(robot.y, gap)
            newang = normal(robot.ang, gap)
            prev = len(robots)
            st.addrobot(st.Robot(newx, newy, newang), robots)
            changed_size = prev < len(robots)
    return st.Particle(robots)
