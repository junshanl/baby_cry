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

import collections

l = collections.deque(maxlen = 3)
l.append(1)
l.append(2)
l.append(3)
l.append(4)

print l

import numpy as np

l = np.empty(10, dtype=np.float32)

print l

import Queue

q = Queue.Queue()

def delete_nth(d, n):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)
