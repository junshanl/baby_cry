import sys
import numpy as np

from core import expectation, maximization, init, adaption


EPSLION = 1e-6

def train_ubm(X, n_components):
    w, mean, cov = init(X, n_components)
    pre_llh = sys.float_info.min 
    pp, llh = expectation(X, w, mean, cov) 
   
    while np.abs(llh - pre_llh) >= EPSLION:
        pre_llh = llh
        w, mean, cov = maximization(X, pp)
        pp, llh = expectation(X, w, mean, cov)
  
    return w, mean, cov


'''
X1 = np.random.multivariate_normal([10,1], [[20,1],[1,20]], 200)
X2 = np.random.multivariate_normal([1,10], [[20,1],[1,20]], 100)

X = np.concatenate((X1, X2))


print train_ubm(X, 2)
'''
