import numpy as np
import yaml
from collections import deque

class Processor():

    def __init__(self, config_path='bin/config.yml', func=None):
        f = open(config_path, 'rb')
        conf = yaml.load(f)
        
        frame_size = conf['model']['frame_size']
        sr = conf['model']['sample_rate']
        record_time = conf['model']['time']
        self.hop_size = conf['model']['hop_size']
        interval = conf['sample']['interval']

        n_cell = 1 + int((sr * record_time - frame_size) / self.hop_size)
        
        self.pool = deque(maxlen=n_cell)
        self.buffer = np.array([])
        self.cur_cell = np.zeros(shape=(frame_size,))

        self.count = interval * sr / self.hop_size 
        self.count_down = self.count

        self.func = func


    def handle(self, data, **wargs):
        data = np.concatenate((self.buffer, np.array(data)))
        size = np.shape(data)[0]

        if size > self.hop_size:
            n_hop = size / self.hop_size
            input_size = n_hop * self.hop_size                  
            self.buffer = data[input_size:]                 
            input_data = np.reshape(data[:input_size], (n_hop,self.hop_size))
            for hop in input_data:
                self.__fill_up(hop)
            self.__activate(n_hop)
        else:
            self.buffer = data


    def __fill_up(self, hop):
        if np.shape(hop)[0] != self.hop_size:
            raise ValueError('hop size mismatch')
        tmp = np.concatenate((self.cur_cell[self.hop_size:], hop))
        self.cur_cell = tmp
        self.pool.append(tmp)


    def __activate(self, n_hop):
        self.count_down = self.count_down - n_hop
        if self.count_down <= 0 and len(self.pool) == self.pool.maxlen:
            self.func(np.array(self.pool))
            self.count_down = self.count

