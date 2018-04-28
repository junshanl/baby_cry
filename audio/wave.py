import struct
import sys
from chunk import Chunk
import __builtin__

__all__ = ["open"]

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003

_array_fmts = None, 'b', 'h', None, 'l'


class Error(Exception):
    pass


class WaveReader:

    def setnchannels(self, nchannels):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if nchannels < 1:
            raise Error, 'bad # of channels'
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error, 'number of channels not set'
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if sampwidth < 1 or sampwidth > 4:
            raise Error, 'bad sample width'
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._sampwidth:
            raise Error, 'sample width not set'
        return self._sampwidth

    def setframerate(self, framerate):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if framerate <= 0:
            raise Error, 'bad frame rate'
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error, 'frame rate not set'
        return self._framerate

    def __init__(self, f):
        f = __builtin__.open(f, 'rb')
        self._i_opened_the_file = f
        try:
            self.initfp(f)
        except:
            f.close()
            raise

    def initfp(self, file):
        self._file = Chunk(file, bigendian=0)
        if self._file.getname() != 'RIFF':
            raise Error, 'file does not start with RIFF id'
        if self._file.read(4) != 'WAVE':
            raise Error, 'not a WAVE file'
        self._fmt_chunk_read = 0
        while 1:
            try:
                chunk = Chunk(self._file, bigendian=0)
            except EOFError:
                break
            chunkname = chunk.getname()
            if chunkname == "fmt ":
                self.read_fmt_chunk(chunk)
                self._fmt_chunk_read = 1
            if chunkname == "data":
                if not self._fmt_chunk_read:
                    raise Error, 'data chunk before fmt chunk'
                self._data_chunk = chunk
                self._nframes = chunk.chunksize // self._framesize
                break
            chunk.skip()
        if not self._fmt_chunk_read or not self._data_chunk:
            raise Error, 'fmt chunk and/or data chunk missing'

    def readFrames(self, nframes):
        if nframes == 0:
            return ''

        if self._sampwidth in (2, 4) and sys.byteorder == 'big':
            # unfortunately the fromfile() method does not take
            # something that only looks like a file object, so
            # we have to reach into the innards of the chunk object
            import array
            chunk = self._data_chunk
            data = array.array(_array_fmts[self._sampwidth])
            assert data.itemsize == self._sampwidth
            nitems = nframes * self._nchannels
            if nitems * self._sampwidth > chunk.chunksize - chunk.size_read:
                nitems = (chunk.chunksize - chunk.size_read) // self._sampwidth
            data.fromfile(chunk.file.file, nitems)
            # "tell" data chunk how much was read
            chunk.size_read = chunk.size_read + nitems * self._sampwidth
            # do the same for the outermost chunk
            chunk = chunk.file
            chunk.size_read = chunk.size_read + nitems * self._sampwidth
            data.byteswap()
            data = data.tostring()
        else:
            data = self._data_chunk.read(nframes * self._framesize)
        return data

    def read_fmt_chunk(self, chunk):
        wFormatTag, self._nchannels, self._framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack(
            '<HHLLH', chunk.read(14))
        if wFormatTag == WAVE_FORMAT_PCM or wFormatTag == WAVE_FORMAT_IEEE_FLOAT:
            sampwidth = struct.unpack('<H', chunk.read(2))[0]
            self._sampwidth = (sampwidth + 7) // 8
        else:
            raise Error, 'unknown format: %r' % (wFormatTag,)
        self._framesize = self._nchannels * self._sampwidth
        self._comptype = 'NONE'
        self._compname = 'not compressed'

    def close(self):
        self._i_opened_the_file.close()


class WaveWriter:

    def __init__(self, f):
        self._i_opened_the_file = None
        if isinstance(f, basestring):
            f = __builtin__.open(f, 'wb')
            self._i_opened_the_file = f
        try:
            self.initfp(f)
        except:
            if self._i_opened_the_file:
                f.close()
            raise

    def initfp(self, file):
        self._file = file
        self._convert = None
        self._nchannels = 0
        self._sampwidth = 0
        self._framerate = 0
        self._nframeswritten = 0
        self._datawritten = 0
        self._headerwritten = False
        self._format = WAVE_FORMAT_PCM
        self._nframes = 0

    def setfomat(self, format):
        if self._datawritten:
            raise Error, 'cannot change format after writing'
        if format not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
            raise Error, 'unknown format'
        self._format = format

    def setnchannels(self, nchannels):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if nchannels < 1:
            raise Error, 'bad # of channels'
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error, 'number of channels not set'
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if sampwidth < 1 or sampwidth > 4:
            raise Error, 'bad sample width'
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._sampwidth:
            raise Error, 'sample width not set'
        return self._sampwidth

    def setframerate(self, framerate):
        if self._datawritten:
            raise Error, 'cannot change parameters after starting to write'
        if framerate <= 0:
            raise Error, 'bad frame rate'
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error, 'frame rate not set'
        return self._framerate

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        self._ensure_header_written(len(data))
        nframes = len(data) // (self._sampwidth * self._nchannels)
        if self._convert:
            data = self._convert(data)
        if self._sampwidth in (2, 4) and sys.byteorder == 'big':
            import array
            a = array.array(_array_fmts[self._sampwidth])
            a.fromstring(data)
            data = a
            assert data.itemsize == self._sampwidth
            data.byteswap()
            data.tofile(self._file)
            self._datawritten = self._datawritten + len(data) * self._sampwidth
        else:
            self._file.write(data)
            self._datawritten = self._datawritten + len(data)
        self._nframeswritten = self._nframeswritten + nframes

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        try:
            if self._file:
                self._ensure_header_written(0)
                if self._datalength != self._datawritten:
                    self._patchheader()
                self._file.flush()
        finally:
            self._file = None
            file = self._i_opened_the_file
            if file:
                self._i_opened_the_file = None
                file.close()

    def _ensure_header_written(self, datasize):
        if not self._headerwritten:
            if not self._nchannels:
                raise Error, '# channels not specified'
            if not self._sampwidth:
                raise Error, 'sample width not specified'
            if not self._framerate:
                raise Error, 'sampling rate not specified'
            self._write_header(datasize)

    def _write_header(self, initlength):
        assert not self._headerwritten
        self._file.write('RIFF')
        if not self._nframes:
            self._nframes = initlength / (self._nchannels * self._sampwidth)
        self._datalength = self._nframes * self._nchannels * self._sampwidth
        self._form_length_pos = self._file.tell()
        self._file.write(struct.pack('<L4s4sLHHLLHH4s',
                                     36 + self._datalength, 'WAVE', 'fmt ', 16,
                                     self._format, self._nchannels, self._framerate,
                                     self._nchannels * self._framerate * self._sampwidth,
                                     self._nchannels * self._sampwidth,
                                     self._sampwidth * 8, 'data'))
        self._data_length_pos = self._file.tell()
        self._file.write(struct.pack('<L', self._datalength))
        self._headerwritten = True

    def _patchheader(self):
        assert self._headerwritten
        if self._datawritten == self._datalength:
            return
        curpos = self._file.tell()
        self._file.seek(self._form_length_pos, 0)
        self._file.write(struct.pack('<L', 36 + self._datawritten))
        self._file.seek(self._data_length_pos, 0)
        self._file.write(struct.pack('<L', self._datawritten))
        self._file.seek(curpos, 0)
        self._datalength = self._datawritten


def open(f, mode=None):
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return WaveReader(f)
    elif mode in ('w', 'wb'):
        return WaveWriter(f)
    else:
        raise Error, "mode must be 'r', 'rb', 'w', or 'wb'"


if __name__ == "__main__":
    wav = open('1.wav', 'r')
    wav2 = open('test.wav', 'w')

    wav2.setfomat(WAVE_FORMAT_IEEE_FLOAT)
    wav2.setframerate(44100)
    wav2.setnchannels(1)
    wav2.setsampwidth(4)

    data = wav.readFrames(220000)
    wav2.writeframes(data)
    wav2.close()
