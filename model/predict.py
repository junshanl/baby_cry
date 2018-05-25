from keras.models import load_model
from scipy.io import wavfile
from audio.signals import mel_spectrogram, power_to_db

import numpy as np

model = load_model('esc50_vgg16_stft_weights_train_last_2_base_layers.best.hdf5')

sr, data = wavfile.read('./ESC-50-master/audio/1-100038-A-14.wav')
sr, data = wavfile.read('new.wav')

mel_db = power_to_db(mel_spectrogram(data, None,sr))

slot = mel_db[:, 50:477]

height, width = np.shape(slot) 
slot = np.reshape(slot, (1, height, width, 1))

print np.shape(slot)
print model.predict(slot).argmax(axis =1)
print model.predict(slot)




