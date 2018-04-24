import pyaudio
import matplotlib
import struct

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import wave

FORMAT = pyaudio.paFloat32
SAMPLEFREQ = 44100
FRAMESIZE = 1024
NOFFRAMES = 220
p = pyaudio.PyAudio()
print('running')

stream = p.open(format=FORMAT,channels=1,rate=SAMPLEFREQ,input=True,frames_per_buffer=FRAMESIZE)
data = stream.read(NOFFRAMES*FRAMESIZE)
decoded = struct.unpack(str(NOFFRAMES*FRAMESIZE)+'f',data)

fig = plt.figure()
# stop Recording
plt.plot(decoded)

fig.savefig("foo.png")

stream.stop_stream()
stream.close()
p.terminate()
