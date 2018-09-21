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
_, _, ypred, cov = sid.identify_armax(u, y, 2, 3, conf=1000)
ypred = append([], ypred)

ndots = 5
yaux = append([], y)
y10 = [yaux[i * ndots] for i in xrange(len(yaux) / ndots)]

plt.figure(1)
plt.plot(range(0, 2500, ndots), y10, 'bo', markersize=3)
plt.plot(ypred, 'k--', linewidth=1)

# plt.subplot(212)
# plt.plot(range(0, 500, ndots), y10, 'r^', markersize=3, linewidth=.6)
# plt.plot(ypred2delay[:500], 'k:', linewidth=1)

# p1 = plt.plot(range(0, 1000, ndots), y10, 'ro', markersize=3, linewidth=.6)
# p4 = plt.plot(ypred, 'k:', linewidth=1)
plt.xlabel('Quantidade de amostras')
plt.show()
