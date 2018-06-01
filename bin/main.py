from audio_processor import Processor
from player import Player
from Recorder import Recorder
import threading
import time
import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from keras.models import load_model
from audio.signals import mel_spectrogram, power_to_db


model = load_model('esc50_vgg16_stft_weights_train_last_2_base_layers.best.hdf5')
graph = tf.get_default_graph()

def thread_method(data):   
    x = power_to_db(mel_spectrogram(None, S=data))    
    height, width = np.shape(x)
    x = np.reshape(x, (1, height, width, 1))
    global graph
    with graph.as_default():
        print model.predict(x)[0][20]


def predict(data): 
    t = threading.Thread(target=thread_method, args=(data, ))
    t.start()

proc = Processor(func=predict)
recorder = Recorder(proc.handle)
recorder.recorder()


'''
player = Player('./t.wav', None)
player.play()
'''
