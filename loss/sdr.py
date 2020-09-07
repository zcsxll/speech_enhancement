import os
import sys
import torch

sys.path.append(os.path.join(os.path.abspath('.'), 'util'))
import stft

def corr(x, y, eps=1e-8):
    return torch.sum(x * y, dim=-1, keepdim=False) + eps

def compute_sdr(x, y):
    e = x - y
    return -10 * (torch.log10(corr(y, y)) - torch.log10(corr(e, e)))

class Loss(torch.nn.Module):
    def __init__(self, fft_size=512, hop_size=256):
        super(Loss, self).__init__()
        self.stft = stft.STFT(fft_size, hop_size)

    def forward(self, pred_waves, target_waves):
        if target_waves.dtype == torch.int16:
            target_waves = target_waves / 32768.0

        with torch.no_grad():
            target_waves = self.stft.inverse(*self.stft.transform(target_waves))

        sdr = compute_sdr(pred_waves, target_waves).mean()
        return sdr