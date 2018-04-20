import struct
from chunk import Chunk

f = open("1.wav", "rb")
chunk = Chunk(f, bigendian=0)
print f.tell()
print chunk.getname()
print chunk.getsize()
print chunk.read(4)
print f.tell()

chunk = Chunk(f, bigendian=0)
print chunk.getname()
print chunk.getsize()
print f.tell()

wFormatTag, _nchannels, _framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from(
    '<HHLLH', chunk.read(14))


print wFormatTag
print _nchannels
print _framerate
print f.tell()

sampwidth = struct.unpack_from('<H', chunk.read(2))[0]
sampwidth = (sampwidth + 7) // 8

print sampwidth
