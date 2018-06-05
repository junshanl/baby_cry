import yaml
from core import adaption, expectation
from audio.timit_reader import read_timit
from audio.signals import mfcc 
import numpy as np

with open('ubm.yml') as f:
    m = yaml.load(f)

c = m['cov']
w = m['w']
m = m['mean']

path = './TIMIT/TRAIN/DR1/FDAW0/SA1.WAV'

sr, data = read_timit(path)
mfcc_feat = mfcc(data, sr, int(sr*0.025), int(sr*0.01))

X = mfcc_feat.T


for i in range(10):
    pp, llh = expectation(X, w, m, c)
    w, m, c = adaption(X, pp, w, m, c)
    
