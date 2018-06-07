from ubm import *
import numpy as np

path = './TIMIT/TRAIN/*/*/*.WAV'
X = preprocess(path)
X = X[:400000]

train_ubm(X, 1024, save_path='./', save_name='ubm')

    
