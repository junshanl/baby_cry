"""PyAudio Example: Play a WAVE file."""

import pyaudio
from chunk import Chunk
import struct

CHUNK = 1024
f = open("1.wav", "rb")
chunk = Chunk(f, bigendian=0)
chunk.read(4)
chunk = Chunk(f, bigendian=0)

wFormatTag, nchannels, framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from(
    '<HHLLH', chunk.read(14))

sampwidth = struct.unpack_from('<H', chunk.read(2))[0]
sampWidth = (sampwidth + 7) // 8

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(sampwidth),
                channels=nchannels,
                rate=framerate,
                output=True)

data = wf.readframes(CHUNK)

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()
