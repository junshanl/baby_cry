from ubm import *
import numpy as np
from ubm import expectation

path = './TIMIT/TRAIN/DR1/*/*.WAV'
X = preprocess(path)

'''
train_ubm(X, 1024, max_iters=10, save_path='./', save_name='ubm2')
'''

w,m,c =load_ubm('ubm2_1.yml')
train_ubm(X, 1024, weight=w, means=m, covars=c, max_iters=10, save_path='./', save_name='ubm2')
pp, llh = expectation(X, w, m ,c)

print np.argmax(pp, axis=1)
print np.exp(-1e1)
