import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


try:
    import scipy.fft
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft
'''source: https://github.com/matteo-ronchetti/torch-radon'''

    

class AbstractFilter(nn.Module):
    def __init__(self):
        super(AbstractFilter, self).__init__()
    def forward(self, x):
        # print("xtensor shape:", x.shape)
        input_size = x.shape[2]  # Original projection height
        # print("input_height:", input_size)
        # Pad height (dim=2) to next power-of-2 (at least 64)
        projection_size_padded = max(64, int(2 ** (2 * torch.tensor(input_size )).float().log2().ceil()))
        # print("projection_size_padded:", projection_size_padded)
        pad_height = projection_size_padded - input_size
        # Pad bottom in height dimension
        padded_tensor = F.pad(x, (0, 0, 0, pad_height))  # shape: (B, C, projection_size_padded, W)
        # print("Padded tensor shape:", padded_tensor.shape)
        f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
        # print("f shape:", f.shape)
        fourier_filter = self.create_filter(f)
        # print("fourier_filter:", fourier_filter.shape)
        fourier_filter = fourier_filter.view(1, 1, -1, 1)
        # print("fourier_filter_unsqueezed:", fourier_filter.shape)
        # projection = torch.rfft(padded_tensor.transpose(2,3), 1, onesided=False).transpose(2,3) * fourier_filter
        # return torch.irfft(projection.transpose(2,3), 1, onesided=False).transpose(2,3)[:,:,:input_size,:]
        # Apply FFT along the detector axis (dim=2)
        fft_result = torch.fft.fft(padded_tensor, dim=2)  # transpose to apply FFT along the original detector dim
        # fft_result = fft_result.transpose(2, 3)  # revert transpose
        # Apply the filter (broadcasted along batch and angle dimensions)
        filtered_projection = 2 * fft_result.real * fourier_filter  # filter is real, only apply on real part
        # IFFT along detector axis (dim=2), need to re-combine with zeros imaginary part if using only real
        ifft_result = torch.fft.ifft(filtered_projection + 0j, dim=2).real  # add imaginary zero for complex input
        # Crop to original input size
        return ifft_result[:, :, :input_size, :]
    
    

class RampFilter(AbstractFilter):
    def __init__(self):
        super(RampFilter, self).__init__()
    def create_filter(self, f):
        return f

    

