import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from signals import spectrogram, fft_frequencies, mel, mfcc
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.ticker import Formatter, ScalarFormatter
import matplotlib.colors as colors

def __coord_fft_hz(n, sr=22050, **_kwargs):
    '''Get the frequencies for FFT bins'''
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

def __mesh_coords(ax_type, coords, n, **kwargs):
   
   if coords is not None:
        if len(coords) < n:
            raise ParameterError('Coordinate shape mismatch: '
                                 '{}<{}'.format(len(coords), n))
        return coords
   
   coord_map = {'linear': __coord_fft_hz,
                 'hz': __coord_fft_hz,
                 'log': __coord_fft_hz,
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

def specshow(data, x_coords=None, y_coords=None,
             x_axis=None, y_axis=None, **kwargs):
    
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')
    
    all_params = dict(kwargs=kwargs)

    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)


    axes = plt.gca()
    plt.pcolor(x_coords, y_coords, data) 

    return axes



def amplitude_to_db(spec):
    power = amplitude_to_power(spec)
    return 10.0 * np.log10(power)

def amplitude_to_power(spec):
    magnitude = np.abs(spec)
    power = np.square(magnitude)
    return power


import librosa.feature
import librosa.util
import librosa.display

sr ,y = wavfile.read("6.wav")
print np.max(y)
y = librosa.util.normalize(np.array(y, dtype = np.float))
d = spectrogram(y)

fig = plt.figure()
plt.plot(y)
fig.savefig("plot4.png")


print y, sr
fig = plt.figure()
mel_bank = mel(44100, 2048)

D = librosa.feature.mfcc(y, sr)
print np.shape(D)
librosa.display.specshow(np.transpose(D)) 

fig.savefig('plot7.png')

fig = plt.figure()

print np.shape(mfcc(y, 44100, 2048))
librosa.display.specshow(mfcc(y, 44100, 2048)) 
fig.savefig('plot5.png')





fig = plt.figure()
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
fig.savefig('plot6.png')




