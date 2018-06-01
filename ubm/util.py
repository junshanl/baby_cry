from audio.signals import mfcc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audio.timit_reader import read_timit
from audio.display import specshow
from scipy.io import wavfile
import numpy as np
import librosa
import glob
from ubm import train_ubm

path = './TIMIT/TRAIN/DR*/*/*.WAV'

def read(path):
    wav_path = glob.glob(path)
   
    feat = [] 
    for p in wav_path:
       sr, data = read_timit(wav_path[0])
       mfcc_feat = mfcc(data, sr, int(sr*0.025), int(sr*0.01))
       feat.append(mfcc_feat.T)
       print p 
    return feat   

data = np.array(read(path))
print np.shape(data)
data = np.reshape(data, (-1, 20))
   
train_ubm(data.T, 32)
