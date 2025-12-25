import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    import scipy.fft
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = np.fft

PI = np.pi

class AbstractFilter(nn.Module):
    """
    Apply 1D frequency filter along detector axis (H, dim=2) of a sinogram.
    Proper complex-domain multiply; crop back to original size.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 << (int(n - 1).bit_length())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] (H = detector)
        H = int(x.shape[2])
        Hpad = max(64, self._next_pow2(H))
        pad_h = Hpad - H

        x_pad = F.pad(x, (0, 0, 0, pad_h))  # [B,C,Hpad,W]

        # build filter of length Hpad (on same device/dtype)
        f = self._get_fourier_filter(Hpad).to(x_pad.device, dtype=x_pad.dtype)  # [Hpad]
        f = self.create_filter(f).view(1, 1, -1, 1)                             # [1,1,Hpad,1]

        # complex fft along detector axis
        X = torch.fft.fft(x_pad, dim=2)                        # complex64/128
        # broadcast real filter to complex dtype
        f_c = f.to(X.dtype)
        Y = X * f_c                                           # complex multiply
        y = torch.fft.ifft(Y, dim=2).real                     # back to real
        return y[:, :, :H, :]

    def _get_fourier_filter(self, size: int) -> torch.Tensor:
        # classic ramp via analytic Fourier transform of |ω|
        # build time-domain kernel f[n] whose FFT is the ramp
        n1 = np.arange(1, size//2 + 1, 2)
        n2 = np.arange(size//2 - 1, 0, -2)
        n = np.concatenate([n1, n2]).astype(np.float64)

        f = np.zeros(size, dtype=np.float64)
        f[0] = 0.25
        f[1::2] = -1.0 / (PI * n) ** 2

        F = 2.0 * np.real(fftmodule.fft(f))  # ramp spectrum
        return torch.from_numpy(F.astype(np.float32))

    def create_filter(self, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class RampFilter(AbstractFilter):
    def create_filter(self, f: torch.Tensor) -> torch.Tensor:
        return f

class HannFilter(AbstractFilter):
    def create_filter(self, f: torch.Tensor) -> torch.Tensor:
        n = torch.arange(0, f.shape[0], device=f.device, dtype=f.dtype)
        hann = 0.5 - 0.5 * torch.cos(2.0 * torch.tensor(PI, dtype=f.dtype, device=f.device) * n / (f.shape[0] - 1))
        hann = torch.roll(hann, shifts=hann.shape[0] // 2, dims=0)
        return f * hann
