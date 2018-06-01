from wave import WaveWriter, WAVE_FORMAT_IEEE_FLOAT
import struct
import numpy as np


def read_timit(path):
    f = open(path, 'r')
    b = f.read(1024)
    info = b.decode("utf-8").split('\n')

    data = []

    d = {'sample_count':None,
           'sample_rate':None,
           'sample_n_bytes':None}

    for i in info:
        para = i.split(' ')
        if d.has_key(para[0]):
            d[para[0]] = int(para[2])

    raw_data = f.read(d['sample_count'] * d['sample_n_bytes'])

    for i, k in zip(raw_data[0::2], raw_data[1::2]):
       data.append(struct.unpack('<h', ''.join((i,k)))[0])

    data = np.array(data, dtype=np.float32) / np.max(data)

    return d['sample_rate'], data

'''
f = open("./TIMIT/TRAIN/DR1/FCJF0/SA1.WAV", 'rb')
from scipy.io import wavfile
print d['sample_rate']
wavfile.write('t.wav', d['sample_rate'], np.array(data, dtype=np.float32))
sr, data = wavfile.read('t.wav')

print data


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(data)
fig.savefig('1.png')
'''

