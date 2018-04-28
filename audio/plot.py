import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
 

sr, data = wavfile.read("1.wav")

print len(data)

fig = plt.figure()

plt.plot(data)

fig.savefig('plot2.png')
