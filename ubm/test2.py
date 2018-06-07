from ubm import *
import numpy as np
from ubm import expectation
 
path = './TIMIT/TRAIN/DR1/*/*.WAV'
X = preprocess(path)
X = X[:400000]

train_ubm(X, 1024, max_iter=10, save_path='./', save_name='ubm2')
    
