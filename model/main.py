import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import csv
import os
from scipy.io import wavfile
from audio.signals import mel_spectrogram, power_to_db
from audio.display import specshow

csvfile = open('ESC-50-master/meta/esc50.csv' ,'rb')
lines = csv.reader(csvfile)
for line in list(lines)[1:]:
    file_name = line[0]
    file_path = os.path.join('ESC-50-master/audio', file_name)
    if os.path.exists(file_path):
        sr, y = wavfile.read(file_path)
        mel_spec_power = mel_spectrogram(y, sr)
        save_dir = os.path.join('ESC-50-master/spectrogram', line[3])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = "".join((os.path.splitext(file_name)[0],'.png'))   
        save_path = os.path.join(save_dir, save_name)
        
        fig = plt.figure()
        specshow(power_to_db(mel_spec_power), x_axis = 'time', y_axis = 'hz', sr = sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(file_name)
        fig.savefig(save_path)
        plt.close()
