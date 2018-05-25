import pyaudio
import struct

from audio import wave

class Player:

    def __init__(self, audio_path, handler = None, frame_size = 2048):
        self.wave = wave.open(audio_path, 'r')
        self.path = audio_path
        self._samplewidth = self.wave.getsampwidth()
        self._framerate = self.wave.getframerate()
        self._nchannels = self.wave.getnchannels()
        self._handler = handler
        self.frame_size = frame_size
    
    def stream_callback(self, in_data, frame_count, time_info, status):
        frames = self.wave.readFrames(frame_count)
        if len(frames) > 0:
            data = struct.unpack(str(len(frames) / self._samplewidth) + 'f', frames)
            if self._handler != None:
                self._handler(list(data))
            status = pyaudio.paContinue
        else:
            status = pyaudio.paComplete
        return (frames, status)
    
    def play(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(self._samplewidth),
                        channels=self._nchannels,
                        rate=self._framerate,
                        output=True,
                        stream_callback=self.stream_callback)

        stream.start_stream()
        print "start playing {}".format(self.path) 

        try:
            while stream.is_active():
                pass
        except KeyboardInterrupt:
            self.close(stream)

        print "finish playing {}".format(self.path)
        self.close(stream)


    def close(self, stream):
        stream.stop_stream()
        stream.close()
        self.wave.close()

