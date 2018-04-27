import pyaudio
import time
import numpy as np
import scipy.signal as signal
WIDTH = 2
CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()
b,a=signal.iirdesign(0.03,0.07,5,40)
fulldata = np.array([])

import wave

wf = wave.open("record.wav", "wb")
wf.setnchannels(1)
wf.setsampwidth(4)
wf.setframerate(44100)

def callback(in_data, frame_count, time_info, status):
    print frame_count
    wf.writeframes(in_data)
    return (in_data, pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                output=True,
                input=True,
                frames_per_buffer = 1024,
                stream_callback=callback)

stream.start_stream()

import time
time.sleep(5)

stream.stop_stream()
stream.close()
wf.close()
p.terminate()
