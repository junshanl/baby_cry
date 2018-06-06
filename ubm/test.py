import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from core import expectation, expectation_old, maximization, adaption, init

from scipy.stats import multivariate_normal

a = np.array([1,2,3])

X1 = np.random.multivariate_normal([10,1], [[2,1],[1,2]], 10)
X2 = np.random.multivariate_normal([1,10], [[2,1],[1,2]], 10)
X3 = np.random.multivariate_normal([5,10], [[2,1],[1,2]], 10)


X = np.concatenate((X1, X2))


w,m,c = init(X, 2)
for i in range(20):
    pp, llh = expectation(X, w, m, c)
    print llh 
    w, m, c = maximization(X, pp)


X = np.concatenate((X, X3))
'''
for i in range(10):
    pp, llh = expectation(X, w, m, c)
    print llh 
    pp, llh = expectation_old(X, w, m, c)
    print llh
    w, m, c = adaption(X, pp, w, m, c)

print w
print m
print c
'''

x = np.linspace(-2., 15.)
y = np.linspace(-2., 15.)

X = np.concatenate((X, X3))
xx, yy = np.meshgrid(x, y)
xy = np.array([xx.ravel(), yy.ravel()]).T


Z = [-expectation(np.array([cell]),w,m,c)[1] for cell in xy]
print (expectation([[5,5]],w,m,c))
print (expectation([[1,10]],w,m,c))
print np.log(multivariate_normal.pdf([[5,5]],m[0],c[0])) + np.log(multivariate_normal.pdf([[5, 5]],m[1],c[1])) 
print np.log(multivariate_normal.pdf([[1,10]],m[0],c[0])) + np.log(multivariate_normal.pdf([[1, 10]],m[1],c[1]))
fig = plt.figure()
plt.contour(xx, yy, np.reshape(Z,(50,50)))
plt.scatter(X[:, 0], X[:, 1], .8)
fig.savefig("1.png")
