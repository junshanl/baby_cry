import numpy as np

from audio.signals import spectrogram, mel
from audio.display import specshow

__all__ = {'extract_spec'}

def extract_spec(y, sr, frame_size = 2048, hop_length = 512, ):
    spec = spectrogram(y, None, frame_size, hop_length)
    mel_bank = mel(sr, frame_size)
    mel_spec = np.dot(mel_bank, spec)
    mel_spec_power = np.square(mel_spec)
    return mel_spec_power






