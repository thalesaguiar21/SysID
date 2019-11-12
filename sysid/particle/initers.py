import sample.particle.structures as st
import sample.particle.sampling as samp
from sample.utils import rad
from random import uniform
from enum import Enum


class Initer(Enum):
    RANDOM = 0
    SUBPARTICLE = 1


def initialise(nparticles, nrobots, measure=None, strategy=Initer.RANDOM):
    if strategy is Initer.RANDOM:
        return _randomparticles(nparticles, nrobots)
    elif strategy is Initer.SUBPARTICLE:
        return _subparticles(nparticles, nrobots, measure)
    else:
        validiniters = [name for name, _ in Initer.__members__.items()]
        raise ValueError('Invalid initer, use {}'.format(validiniters))


def _randomparticles(nparticles, nrobots):
    particles = []
    robots = _random_robots(nparticles * nrobots)
    for i in range(nparticles):
        particle_robots = robots[i * nrobots:nrobots + i * nrobots]
        particles.append(st.Particle(particle_robots))
    return particles


def _random_robots(nrobots):
    robots = []
    while len(robots) != nrobots:
        x = uniform(st.FIELDMIN, st.FIELDMAX)
        y = uniform(st.FIELDMIN, st.FIELDMAX)
        z = rad(uniform(-180, 180))
        st.addrobot(st.Robot(x, y, z), robots)
    return robots


def _generate_static(nrob):
    for i in range(20):
        for j in range(20):
            rx = -0.1 * (i % 10) if i < 10 else 0.1 * (i % 10)
            ry = -0.1 * (j % 10) if j < 10 else 0.1 * (j % 10)
            robots = [st.Robot(rx, ry, ang) for ang in range(-180, 181, 5)]
    particles = []
    for i in range(len(robots)):
        particles.append(st.Particle(robots[i * nrob:nrob + i * nrob]))
    return particles


def _subparticles(nparticles, nrobots, measure):
    statics = _generate_static(nrobots)
    st.evalparticles(statics, measure)
    particles = []
    for i in range(nrobots):
        randoms = _randomparticles(nparticles, nrobots)
        randoms = samp.resample(randoms, nparticles, samp.Sampler.COPY)
        st.evalparticles(randoms, measure)
        sorted(randoms, reverse=True)
        for static in statics:
            if static.weight > randoms[-1].weight:
                _insert(static, randoms)
        particles.extend(randoms)
    return randoms


def _insert(particle, particles):
    for i in range(len(particles)):
        if particle.weight > particles[i].weight:
            particles[i] = particle
            break
