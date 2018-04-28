from chunk import Chunk

f = open("1.wav", "rb")

chunk = Chunk(f, bigendian=0)
chunk.read(4)

chunk = Chunk(f, bigendian=0)

print chunk.chunkname == b"fmt"
print chunk.chunkname == "fmt"
print chunk.chunkname, chunk.chunksize
chunk.read(18)

chunk = Chunk(f, bigendian=0)
print chunk.chunkname, chunk.chunksize
chunk.read(4)

chunk = Chunk(f, bigendian=0)
print chunk.chunkname, chunk.chunksize
print chunk.chunkname == "data"
print chunk.read(882000)[0:10]
print chunk.read(1024)[0:10]
print chunk.tell()


import pyaudio

p = pyaudio.PyAudio()

print pyaudio.paInt16
print pyaudio.paInt32
print pyaudio.paFloat32

