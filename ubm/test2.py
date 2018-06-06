from ubm import *
import numpy as np

path = './TIMIT/TRAIN/DR1/FDAW0/*.WAV'
X = preprocess(path)

train_ubm(X, 1024, save_path='.', save_name='ubm.yaml')

    
