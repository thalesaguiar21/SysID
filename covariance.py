from data import utils as dut
from numpy import append
import matplotlib.pyplot as plt
import sample.identification as sid
import pdb
import sample.metrics as met

'''
ARX = RED
ARMAX = BLUE
'''

u, y = dut.r_dots('tank1.dat', 0, 2, '\t')
_, res10, _, c10 = sid.identify_arx(u, y, 2, 3, conf=10)
_, res100, _, c100 = sid.identify_arx(u, y, 2, 3, conf=100)
_, res1000, _, c1000 = sid.identify_arx(u, y, 2, 3, conf=1000)

c10 = sum([c10[i, i] for i in range(c10.shape[0])]) / c10.shape[0]
c100 = sum([c100[i, i] for i in range(c100.shape[0])]) / c100.shape[0]
c1000 = sum([c1000[i, i] for i in range(c1000.shape[0])]) / c1000.shape[0]

print '10\t& {:<.3E}\t& {:<.3E} \\\\'.format(met.stdev(res10), c10)
print '100\t& {:<.3E}\t& {:<.3E} \\\\'.format(met.stdev(res100), c100)
print '1000\t& {:<.3E}\t& {:<.3E} \\\\'.format(met.stdev(res1000), c1000)

# print '{:<.3E}\t{:<.3E}\t{:<.3E}'.format(
#     met.stdev(res10), met.stdev(res100), met.stdev(res1000))
# print '{:<.3E}\t{:<.3E}\t{:<.3E}'.format(
#     c10, c100, c1000)
