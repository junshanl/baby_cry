import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from audio.signals import mel_spectrogram, power_to_db
from audio.display import specshow

sr, data = wavfile.read('1.wav')
sr, data1 = wavfile.read("./ESC-50-master/audio/1-100032-A-0.wav")
print np.max(data), np.min(data)
print np.max(data1), np.min(data1)

fig = plt.figure()

specshow(power_to_db(mel_spectrogram(data, sr)), x_axis = 'time', y_axis = 'hz', sr = sr)

plt.colorbar(format='%+2.0f dB')

fig.savefig('p.png')

