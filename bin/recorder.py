import pyaudio
import struct

from audio import wave

class Recorder():

    def __init__(self, handler,sample_rate=44100, frame_size=2048, n_channels=1, sample_width=4):
        self._samplewidth = sample_width
        self._framerate = sample_rate
        self._nchannels = n_channels 
        self._handler = handler
        self.frame_size = frame_size
        self.stoped = False

    def stream_callback(self, in_data, frame_count, time_info, status):
        data = struct.unpack(str(frame_count) + 'f', in_data)
        self._handler(list(data))
        status = pyaudio.paContinue
            
        if self.stoped:
            status = pyaudio.paComplete

        return (in_data, status)
 
    def recorder(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(self._samplewidth),
                        channels=self._nchannels,
                        rate=self._framerate,
                        input=True,
                        stream_callback=self.stream_callback)

        stream.start_stream()
        print "start recording ..." 

        try:
            while stream.is_active():
                pass
        except KeyboardInterrupt:
            self.stoped = True
            "finish recording ..."
        self.close(stream)


    def close(self, stream):
        stream.stop_stream()
        stream.close()
