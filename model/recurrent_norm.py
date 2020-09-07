import os
import sys
import torch

sys.path.append(os.path.join(os.path.abspath('.'), 'util'))
import stft

class RNNWrapper(torch.nn.Module):
    def __init__(self, rnn):
        super(RNNWrapper, self).__init__()
        self.rnn = rnn

    def forward(self, x):
        if callable(getattr(self.rnn, 'flatten_parameters', None)):
            self.rnn.flatten_parameters()
        return self.rnn(x)[0]

class Model(torch.nn.Module):
    def __init__(self,
                fft_size=512,
                hop_size=256,
                rnn_cell_type='GRU',
                norm_rnn_cell_size=16,
                norm_rnn_layers=2,
                rnn_cell_size=64,
                rnn_layers=2,):
        super(Model, self).__init__()

        self.stft = stft.STFT(fft_size, hop_size)
        if rnn_cell_type == 'GRU':
            RNN = torch.nn.GRU
        elif rnn_cell_type == 'LSTM':
            RNN = torch.nn.LSTM
        else:
            raise NotImplementedError

        rnns = []
        rnns += [torch.nn.Linear(in_features=fft_size//2, out_features=norm_rnn_cell_size)]
        rnns += [torch.nn.ReLU()]
        rnns += [RNNWrapper(RNN(input_size=norm_rnn_cell_size,
                                hidden_size=norm_rnn_cell_size,
                                num_layers=norm_rnn_layers,
                                batch_first=True))]
        rnns += [torch.nn.Linear(in_features=norm_rnn_cell_size, out_features=fft_size)]
        self.norm_rnn = torch.nn.Sequential(*rnns)

        rnns = []
        rnns += [torch.nn.Linear(in_features=fft_size//2, out_features=rnn_cell_size)]
        rnns += [torch.nn.ReLU()]
        rnns += [RNNWrapper(RNN(input_size=rnn_cell_size,
                                hidden_size=rnn_cell_size,
                                num_layers=rnn_layers,
                                batch_first=True))]
        rnns += [torch.nn.Linear(in_features=rnn_cell_size, out_features=fft_size//2)]
        self.rnn = torch.nn.Sequential(*rnns)

    def total_parameter(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        if x.dtype == torch.int16:
            x = x / 32767.0

        with torch.no_grad():
            re, im = self.stft.transform(x)
            mag = (re**2 + im**2 + 1e-20)**0.5
        
        mean, logstd = self.norm_rnn(mag).chunk(2, dim=-1)
        norm_mag = (mag - mean) / torch.exp(logstd)

        mask = self.rnn(norm_mag)
        re = re * mask
        im = im * mask
        wav = self.stft.inverse(re, im)
        return wav, {'real':re, 'imag':im}

if __name__ == '__main__':
    model = Model()
    print('%fMB' % (model.total_parameter() * 4 / 1024 / 1024))
    pcm = torch.ones(2, 16000 * 5, dtype=torch.int16)
    wav, _ = model(pcm)
    print(wav.shape)