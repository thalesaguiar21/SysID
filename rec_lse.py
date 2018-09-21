from sample.estimation import recursive_lse
import numpy as np

coef = np.matrix('2 1 5; 1 3 4; 0 5 -1; -1 2 3')
rs = np.matrix('1; -7; -15; -8')
params, cov_matrix, vari = recursive_lse(coef, rs, conf=10000)
result = np.dot(coef, params)
result = [e[0, 0] for e in result]
expected = [0.9994501300038658, -6.99994998163175,
            -14.999699950935845, -7.999050253967136]


print 'Parameters'
print '-' * 50

print params

print
print 'Covariance'
print '-' * 50

print cov_matrix

print
print 'Result'
print '-' * 50

print result

print
print 'Residue'
print '-' * 50

print np.matrix(result - rs).T

print
print 'Params variation'
print '-' * 50

print vari
