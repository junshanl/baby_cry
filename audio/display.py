import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from signals import spectrogram, fft_frequencies, mel 
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.ticker import Formatter, ScalarFormatter
import matplotlib.colors as colors

def __coord_hz(n, sr, **_kwargs):
    '''Get the frequencies for FFT bins'''
    if sr == None:
        raise ParameterError('sample rate is not defined')
    
    n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fft_frequencies(sr=sr, n_fft=n_fft)
    fmax = basis[-1]
    basis -= 0.5 * (basis[1] - basis[0])
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis

def __coord_n(n, **_kwargs):
    '''Get bare positions'''
    return np.arange(n+1)


def __mesh_coords(ax_type, n, **kwargs):
   
   coord_map = {'linear': __coord_hz,
                'hz': __coord_hz,
                'time':__coord_n,
                None: __coord_n}
   
   if ax_type not in coord_map:
        raise ParameterError('Unknown axis type: {}'.format(ax_type))

   return coord_map[ax_type](n, **kwargs)

def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return get_cmap(cmap_bool)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        return get_cmap(cmap_seq)
   
    return get_cmap(cmap_div)

def specshow(data, x_axis=None, y_axis=None, sr = None, **kwargs):
    kwargs.setdefault('cmap',get_cmap('magma'))
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')
   
    all_params = dict(kwargs = kwargs, sr = sr)

    y_coords = __mesh_coords(y_axis, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, data.shape[1], **all_params)


    axes = plt.gca()
    img = axes.pcolormesh(x_coords, y_coords, data, **kwargs) 
    plt.sci(img)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    __axis_decorate(axes.xaxis, x_axis)
    __axis_decorate(axes.yaxis, y_axis)
   
    return axes

def __axis_decorate(axis, ax_type):
    if ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')

    if ax_type == 'db':
        axis.set_label_text('decibel')

    if ax_type == 'time':
        axis.set_label_text('time')

from signals import mel_spectrogram 

fig = plt.figure()

sr, y = wavfile.read("1.wav")

mel_spec_power = mel_spectrogram(y, sr)

specshow(10. * np.log10(mel_spec_power), y_axis = 'hz', sr = sr)

import librosa

print np.max(10. * np.log10(mel_spec_power))
print np.max(librosa.power_to_db(librosa.feature.melspectrogram(y, sr)))
plt.colorbar(format='%+2.0f dB')

fig.savefig('plot6.png')





