from sample.particle import filt as pt
from sample.particle import Initer, Sampler
from numpy import loadtxt


fpath = 'examples/robots.txt'
robot_measures = loadtxt(fpath)
pt.filtrateandplot(robot_measures[:, 1:],
                   1200,
                   0.02,
                   Initer.RANDOM,
                   Sampler.COPY)
