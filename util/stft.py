import numpy as np
from scipy.signal import get_window
import torch
import torch.nn.functional as functional


class STFT(torch.nn.Module):
    def __init__(self,
                 fft_size=1024,
                 hop_size=512,
                 win_size=None,
                 window='hamming',
                 dtype='float32'):
        super(STFT, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size

        if win_size is None:
            self.win_size = fft_size
        else:
            self.win_size = win_size

        if not self.win_size % self.hop_size == 0:
            raise ValueError('win_size cannot be divied by hop_size')

        with torch.no_grad():
            ana_win = self.get_ana_win(window, self.win_size)
            syn_win = self.get_syn_win(ana_win)

            pad_size = (self.fft_size - self.win_size) // 2
            ana_win = torch.from_numpy(ana_win)
            ana_win = functional.pad(ana_win, [pad_size, pad_size])
            syn_win = torch.from_numpy(syn_win)
            syn_win = functional.pad(syn_win, [pad_size, pad_size])

            # [num_sampels, num_freq_bins]
            fourier_basis = np.fft.rfft(np.eye(self.fft_size), self.fft_size)
            fourier_basis = np.concatenate(
                [fourier_basis.real, fourier_basis.imag], axis=-1)

            forward_basis = torch.from_numpy(fourier_basis[:, :])
            forward_basis = ana_win[:, None] * forward_basis

            inverse_basis = np.linalg.pinv(fourier_basis[:, :])
            inverse_basis = torch.from_numpy(inverse_basis)
            inverse_basis = syn_win[None, :] * inverse_basis

            if dtype == 'float32':
                self.register_buffer('forward_basis', forward_basis.float())
                self.register_buffer('inverse_basis', inverse_basis.float())
            elif dtype == 'float64':
                self.register_buffer('forward_basis', forward_basis.double())
                self.register_buffer('inverse_basis', inverse_basis.double())
            else:
                raise ValueError('dtype {} is not supported! '.format(dtype))

    def get_ana_win(self, window, Nx):
        win = get_window(window, Nx, fftbins=True)
        return win

    def get_syn_win(self, ana_win):
        num_block = self.win_size // self.hop_size
        ana_win_sq = ana_win * ana_win

        norm_win = []
        for i in range(self.hop_size):
            temp = 0
            for j in range(num_block):
                temp += ana_win_sq[i + j * self.hop_size]
            norm_win += [1 / (temp)]

        syn_win = []
        for i in range(num_block):
            syn_win += norm_win

        syn_win = np.array(syn_win) * ana_win
        return syn_win

    def transform(self, x, keep_dc=False):
        batch_size = None
        if x.dim() == 1:
            # add fake batch dimension
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            batch_size = x.size(0)
        else:
            raise ValueError('input shape is not supported !')

        x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        pad_size = self.win_size - self.hop_size
        x = functional.pad(x, [pad_size, pad_size, 0, 0], mode='reflect')
        x = functional.unfold(x, [1, self.win_size], stride=[1, self.hop_size])

        x = x.transpose(1, 2)
        pad_size = (self.fft_size - self.win_size) // 2
        x = functional.pad(x, [pad_size, pad_size, 0, 0])

        x = torch.einsum('ijk,kl->ijl', x, self.forward_basis)
        cut_off = self.fft_size // 2 + 1

        # split real and imaginary part
        cut_off = self.fft_size // 2 + 1
        re = x[:, :, :cut_off]
        im = x[:, :, cut_off:]

        if not keep_dc:
            re = re[:, :, 1:]
            im = im[:, :, 1:]

        if batch_size is None:
            re = re.squeeze(0)
            im = im.squeeze(0)

        return re, im

    def mag(self, x, keep_dc=False, eps=1e-20):
        re, im = self.transform(x, keep_dc)
        mag = (re**2 + im**2 + eps)**0.5
        return mag

    def mag_phase(self, x, keep_dc=False, eps=1e-20):
        re, im = self.transform(x, keep_dc)
        mag = (re**2 + im**2 + eps)**0.5
        phase = torch.atan2(im, re)
        return mag, phase

    def inverse(self, re, im):
        if re.dim() != im.dim():
            raise ValueError("re's shape and im's shape is not same")

        batch_size = None
        if re.dim() == 2:
            re = re[None, :, :]
            im = im[None, :, :]
        elif re.dim() == 3:
            batch_size = re.size(0)
        else:
            raise ValueError('Expected 2D or 3D tensor, but got {}'.format(
                re.shape))

        if re.size(2) == self.fft_size // 2:
            re = functional.pad(re, [1, 0, 0, 0])
            im = functional.pad(im, [1, 0, 0, 0])

        x = torch.cat([re, im], 2)
        x = torch.einsum('ijk,kl -> ijl', x, self.inverse_basis)

        pad_size = (self.fft_size - self.win_size) // 2
        if pad_size > 0:
            x = x[:, :, pad_size:][:, :, :-pad_size]
        x = x.transpose(1, 2)
        x = functional.fold(
            x,
            [x.size(2) * self.hop_size + self.win_size - self.hop_size, 1],
            [self.win_size, 1],
            stride=[self.hop_size, 1],
        )

        # shape: batch_size, 1, num_samples, 1
        x = x.squeeze(3).squeeze(1)
        pad_size = self.win_size - self.hop_size
        if pad_size > 0:
            x = x[:, pad_size:][:, :-pad_size]

        if batch_size is None:
            x = x.squeeze(0)

        return x

    def mag_phase_inverse(self, mag, phase):
        if mag.shape != phase.shape:
            raise ValueError("mag's shape and phase's shape is not same.")
        re = mag * torch.cos(phase)
        im = mag * torch.sin(phase)
        return self.inverse(re, im)

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     stft = STFT(fft_size=512, hop_size=256, window='hamming')
#     ana_win = stft.get_ana_win('hamming', 512)
#     syn_win = stft.get_syn_win(ana_win)

#     plt.figure()
#     plt.plot(ana_win, color='r')
#     plt.plot(syn_win, color='b')
#     plt.savefig('./out.png')

if __name__ == '__main__':
    import librosa
    import soundfile as sf
    x, sr = sf.read(librosa.util.example_audio_file())
    x = x[:, 0]
    x = x[:len(x) // 2048 * 2048]
    x = x.astype(np.float32)
    wav = x
    print(x.shape)
    torch_stft = STFT(2048, 512, 1024, dtype='float64')
    # xt = torch.from_numpy(x).float()
    # xt = torch.from_numpy(x).double()
    xt = torch.from_numpy(x).float().double()
    # xt = torch.cat([xt[None], xt[None]], 0)
    print('input', xt.shape)
    re, im = torch_stft.transform(xt, True)
    re = re.float().double()
    im = im.float().double()
    re = re[:, 1:]
    im = im[:, 1:]
    wav2 = torch_stft.inverse(re, im)
    print('wav', wav2.shape)
    wav2 = wav2.float()

    spec1 = torch.sqrt(re**2 + im**2)
    spec2 = np.abs(librosa.stft(x, 2048, 512, 1024))
    cspec = librosa.stft(x, 2048, 512, 1024)
    wav1 = librosa.istft(cspec, 512, 1024)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    axs[0].pcolormesh(np.log(spec1.T))
    axs[1].pcolormesh(np.log(spec2))
    fig.savefig('spec.png')

    fig, axs = plt.subplots(5, 1)
    assert isinstance(fig, plt.Figure)
    axs[0].plot(wav1)
    axs[1].plot(wav2)
    print(wav1.shape, wav2.shape)
    axs[2].plot(wav - wav1)
    axs[3].plot(wav - wav2.numpy())
    axs[4].plot(wav1 - wav2.numpy())
    fig.savefig('wav.png')
    sf.write('1.wav', wav1, sr)
    sf.write('2.wav', wav2, sr)
