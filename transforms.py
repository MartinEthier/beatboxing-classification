import random

import numpy as np
import librosa
import torch


class PadToMax(object):
    """
    If waveform is smaller than max length, pad it at both ends with zeros.
    If longer, trim off the end to get it to the right length.

    waveform: (N,) array
    max_length: target length of array
    """
    def __init__(self, max_length):
        assert isinstance(max_length, int)
        self.max_length = max_length

    def __call__(self, waveform):
        wf_size = waveform.shape[0]
        if wf_size > self.max_length:
            # Waveform is longer so trim off the end
            return waveform[:self.max_length]

        if wf_size < self.max_length:
            # Waveform is too short so pad with zeros on both sides randomly
            num_to_pad = self.max_length - wf_size
            num_left = random.randint(0, num_to_pad)
            num_right = num_to_pad - num_left
            return np.pad(waveform, (num_left, num_right), 'constant')

        return waveform


class MelSpectrogram(object):
    """
    Computes a Mel Spectrogram using librosa.
    """
    def __init__(self, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
        assert isinstance(sample_rate, int)
        self.sample_rate = sample_rate
        assert isinstance(n_fft, int)
        self.n_fft = n_fft
        assert isinstance(hop_length, int)
        self.hop_length = hop_length
        assert isinstance(n_mels, int)
        self.n_mels = n_mels

    def __call__(self, waveform):
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec_dB = librosa.power_to_db(mel_spec)
        return mel_spec_dB
    

class MFCC(object):
    """
    Calculates MFCC coefficients from given Mel Spectrogram. If desired, reduces time dimension using
    given numpy function.
    """
    def __init__(self, reduction_func=None):
        self.reduction_func = reduction_func
    
    def __call__(self, spec):
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(S=spec)
        
        # Reduce time dimension if function given
        if self.reduction_func is not None:
            mfcc = np.apply_along_axis(self.reduction_func, 1, mfcc)
        
        return mfcc


class ToTensor(object):
    """
    Normalizes the given spectrogram, adds a single channel dim, and converts it to a torch tensor.
    """
    def __call__(self, spec):
        # Normalize to 0-1 range
        spec -= spec.min()
        spec /= spec.max()

        # Add channel dim
        spec = spec[np.newaxis, :]

        # Convert to torch tensor
        spec_tensor = torch.from_numpy(spec)

        return spec_tensor
