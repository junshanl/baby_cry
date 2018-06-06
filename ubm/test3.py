from core import expectation, expectation_old, maximization, adaption, init, random_cov
import time
import numpy as np
from scipy.stats import multivariate_normal




n_samples = 1000 * 40
clusters = 10
d = 20

cov = np.array([random_cov(d) for i in range(clusters)])
mean = np.random.randn(clusters, d)
w = np.abs(np.random.randn(clusters,))
w = w / np.sum(w)
X = np.empty((0,20), np.float)

print mean  

for i, weight  in enumerate(w):
    print i
    print np.shape(mean[i]), np.shape(cov[i])
    tmp = np.random.multivariate_normal(mean[i], cov[i] , int(n_samples * weight))
    X = np.concatenate((X, tmp))

start = time.time()
pp, llh = expectation(X, w, mean, cov)
middle = time.time()
w2, mean2, cov2 = maximization(X, pp)
end = time.time()
print 'take', middle - start, end - middle

for m1 in mean:
    for m2 in mean2:
       dis = np.linalg.norm(m1 - m2)
       print dis
    print '---------------------------'

