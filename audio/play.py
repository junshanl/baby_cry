"""PyAudio Example: Play a WAVE file."""

import pyaudio
from chunk import Chunk
import struct
from scipy.io import wavfile
import librosa

import pyaudio
import matplotlib
import struct

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import wave


fs, data = wavfile.read('1.wav')
audio_data, sr = librosa.load('1.wav', sr=44100, mono=True, duration=5)
print len(audio_data)

CHUNK = 1024
f = open("1.wav", "rb")
chunk = Chunk(f, bigendian=0)
chunk.read(4)
chunk = Chunk(f, bigendian=0)

wFormatTag, nchannels, framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from(
    '<HHLLH', chunk.read(14))

sampwidth = struct.unpack_from('<H', chunk.read(2))[0]
sampwidth = (sampwidth + 7) // 8


p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(4),
                channels=nchannels,
                rate=framerate,
                output=True)

print stream._frames_per_buffer, stream._rate, stream._format


stream.write(data, 220500)


stream.stop_stream()
stream.close()

p.terminate()

fig = plt.figure()
# stop Recording
plt.plot(data)

fig.savefig("foo.png")
