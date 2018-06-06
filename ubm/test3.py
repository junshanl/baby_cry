from core import expectation, expectation_old, maximization, adaption, init, random_cov
import time
import numpy as np
from scipy.stats import multivariate_normal


n_samples = 1000 * 40
clusters = 1024
d = 20

X = np.random.rand(n_samples, d)
cov = np.array( [random_cov(d) for i in range(clusters)])
mean = np.random.randn(clusters, d)
w = np.abs(np.random.randn(clusters,))
w = w / np.sum(w)

start = time.time()
pp, llh = expectation(X, w, mean, cov)
middle = time.time()
w, mean, cov = maximization(X, pp)
end = time.time()
print 'take', middle - start, end - middle

'''
start = time.time()
_, llh = expectation_old(X, w, mean, cov)
end = time.time()
print llh
print 'take', end - start
print pp
'''
