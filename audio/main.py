import threading
import pyaudio
import numpy as np
import struct
import sys
sys.path.append(".")

from pc_methods.feature_engineer import FeatureEngineer
import os
import pickle
from rpi_methods.baby_cry_predictor import BabyCryPredictor
import time
import numpy as np
from chunk import Chunk
import wave


class Classifer():

    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise Exception("{0} not found".format(model_path))

        with open(model_path, 'rb') as fp:
            self.model = pickle.load(fp)

    def predict(self, data):
        engineer = FeatureEngineer()
        data = np.array(data)
        feat, _ = engineer.feature_engineer(data)
        predictor = BabyCryPredictor(self.model)

        prediction = predictor.predict_proba(feat)

        return prediction


FORMAT = pyaudio.paFloat32
SAMPLERATE = 44100
FRAMESIZE = 1024
NCHANNELS = 1


class Player:

    def __init__(self, audio_path):
        self.wave = wave.open(audio_path, 'r')
        self._samplewidth = self.wave.getsampwidth()
        self._framerate = self.wave.getframerate()
        self._nchannels = self.wave.getnchannels()

    def play(self, listener):
        p = pyaudio.PyAudio()

        def stream_callback(in_data, frame_count, time_info, status):
            frames = self.wave.readFrames(frame_count)
            if len(frames) > 0:
                data = struct.unpack(str(len(frames) / self._samplewidth) + 'f', frames)
                listener.notify(list(data))
                status = pyaudio.paContinue
            else:
                status = pyaudio.paComplete
            return (frames, status)

        stream = p.open(format=p.get_format_from_width(self._samplewidth),
                        channels=self._nchannels,
                        rate=self._framerate,
                        output=True,
                        stream_callback=stream_callback)

        stream.start_stream()

        try:
            while stream.is_active():
                pass
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            self.f.close()
            listener.stop()

        stream.stop_stream()
        stream.close()
        self.wave.close()
        listener.stop()

class Recorder(object):

    def __init__(self):
        self.stoped = False

    def run(self, listener, is_save, file_name):
        p = pyaudio.PyAudio()
        if is_save:
            wf = wave.open(file_name, 'wb')
            wf.setfomat(0x0003)
            wf.setnchannels(NCHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLERATE)

        def stream_callback(in_data, frame_count, time_info, status):
            data = struct.unpack(str(frame_count) + 'f', in_data)
            listener.notify(list(data))
            status = pyaudio.paContinue
            
            if is_save:
                wf.writeframes(in_data)
            
            if self.stoped:
                status = pyaudio.paComplete

            return (in_data, status)

        stream = p.open(format=FORMAT,
                        channels=NCHANNELS,
                        rate=SAMPLERATE,
                        input=True,
                        frames_per_buffer=FRAMESIZE,
                        stream_callback=stream_callback)

        stream.start_stream()

        try:
            while stream.is_active():
                pass
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            listener.stop()
            if is_save:
               wf.close()
            sys.exit()

        stream.stop_stream()
        stream.close()
        listener.stop()
        if is_save:
            wf.close()

    def stop(self):
        self.stoped = True

import threading

class BabyCryDetection(threading.Thread):

    def __init__(self, cls, interval=5, hop=3):
        threading.Thread.__init__(self)
        self.cond = threading.Condition()
        self.cls = cls
        self.frames = []
        self.window = []
        self.interval = interval
        self.hop = hop

    def run(self):
        self.is_active = True

        self.cond.acquire()
        while self.is_active:
            if len(self.window) >= self.interval * SAMPLERATE:
                self.window = [ i / 4 for i in self.window ]
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig = plt.figure()
                plt.plot(self.window)
                fig.savefig('plot1.png')
                print cls.predict(self.window)
                shift = self.hop * (int)(SAMPLERATE / FRAMESIZE) * FRAMESIZE
                del self.window[0:shift]
            self.cond.wait()
        self.cond.release()

    def notify(self, data):
        self.cond.acquire()
        self.window.extend(data)        
        self.frames.extend(data)
        self.cond.notify()
        self.cond.release()

    def stop(self):
        print "stop"
        self.cond.acquire()
        self.is_active = False
        self.cond.notify()
        self.cond.release()

if __name__ == "__main__":

    cls = Classifer("./output/model/model.pkl")
    bcd = BabyCryDetection(cls)
    bcd.start()    
    
    player = Player("new.wav")
    player.play(bcd)
    '''
    recorder = Recorder()
    recorder.run(bcd, True, "record.wav")
    time.sleep(10)
    print "stop"
    recorder.stop()
    '''
