import wave
import struct
from scipy.io import wavfile


sr, data = wavfile.read('record.wav')

data = [ d * 6  for d in data]

data = struct.pack(str(len(data)) + "f", *data)

wav =  wave.open("new.wav", "w")

wav.setfomat(wave.WAVE_FORMAT_IEEE_FLOAT)
wav.setframerate(44100)
wav.setnchannels(1)
wav.setsampwidth(4)

wav.writeframes(data)
wav.close()
