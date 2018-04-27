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


file_name = "record.wav"

CHUNK = 1024
f = open(file_name, "rb")
chunk = Chunk(f, bigendian=0)
print chunk.read(4)
chunk = Chunk(f, bigendian=0)

wFormatTag, nchannels, framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from(
    '<HHLLH', chunk.read(14))

sampwidth = struct.unpack_from('<H', chunk.read(2))[0]
sampwidth = (sampwidth + 7) // 8

chunk.read(2)

print nchannels, framerate, sampwidth

chunk = Chunk(f, bigendian=0)
print chunk.chunkname chunk.chunksize 
chunk.read(4)

chunk = Chunk(f, bigendian=0)
print chunk.chunkname
print chunk.chunkname, chunk.chunksize
data = chunk.read(chunk.chunksize)

'''
p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(sampwidth),
                channels=nchannels,
                rate=framerate,
                output=True)

stream.write(data, 220500)


stream.stop_stream()
stream.close()

p.terminate()
'''
#fig = plt.figure()

# stop Recording
#plt.plot(audio_data)

#fig.savefig("foo.png")
