import numpy as np

from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile
from scipy.signal import get_window, filtfilt 
from scipy.fftpack import fft, dct


def n_frames(y, frames_size, hop_size):
    return 1 + int((len(y) - frame_size) / hop_size)

def enframe(y, frame_length=2048, hop_length=512):
    n_frames = 1 + int((len(y) - frame_length) / hop_length)
    y_frames = as_strided(y, shape=(n_frames, frame_length),
                          strides=(y.itemsize * hop_length, y.itemsize))
    return y_frames


def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def hz_to_mel(frequencies, htk=False):
    frequencies = np.asarray(frequencies)
    
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def mel_frequencies(n_mels, fmin, fmax, htk):

    min_mel = hz_to_mel(fmin, htk)
    max_mel = hz_to_mel(fmax, htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk)


def fft_frequencies(sr, n_fft):
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):
    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights

 
def spectrogram(frames, power = 1):
    n_frame, n_fft = np.shape(frames) 
    n_valid_freqz = int(1 + n_fft // 2)

    fft_window = get_window('hann', n_fft, fftbins=True)
    stft_matrix = np.empty((n_valid_freqz, n_frame), dtype=np.float)

    for i in range(n_frame):
        stft = fft(fft_window * frames[i])[:n_valid_freqz]
        stft_matrix[:, i] = np.abs(stft) ** power 
    
    return stft_matrix

def mel_spectrogram(y, S=None, sr=44100, frame_size=2048, hop_length=512, power=2):
    if S is not None:
        frames = S
    else:
        frames = enframe(y, frame_length=frame_size, hop_length=hop_length)
    spec = spectrogram(frames, power)
    mel_bank = mel(sr, frame_size)
    mel_spec_power = np.dot(mel_bank, spec)
    return mel_spec_power


def mfcc(y, sr, n_mfcc=20, dct_type=2):
    S = mel_spectrogram(y, sr)
    S = power_to_db(S)
    return dct(S, axis=0, type=dct_type)[:n_mfcc]

def power_to_db(S, ref = 1.0, amin = 1e-10, top_db = 80.0 ):
    
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(magphase(D, power=2)[0]) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S
    
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec
