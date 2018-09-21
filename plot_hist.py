from data import utils as dut
from numpy import append, matrix
import matplotlib.pyplot as plt
import sample.identification as sid
import pdb

'''
ARX = RED
ARMAX = BLUE
'''

u, y = dut.r_dots('tank1.dat', 0, 1, '\t')
_, _, _, param_hist = sid.identify_armax(u, y, 1, 0)


param_hist = matrix(param_hist)
# pdb.set_trace()

theta = []
plt.figure(1)
plt.ylabel('Valor do parametro')
plt.xlabel('Tempo')
for i in range(param_hist.shape[1]):
    theta.append(append([], param_hist[:, i]))

# pdb.set_trace()
plt.plot(theta[0], 'r--', linewidth=2)
plt.plot(theta[1], 'b:', linewidth=2)
plt.plot(theta[2], 'g-.', linewidth=2)
plt.show()
