import os
import sys
import torch
import numpy as np
import pypesq
sys.path.append(os.path.join(os.path.abspath('.'), 'util'))
import compute_stoi

class Transform():
    def __init__(self, training=True, hop_size=256):
        self.training = training
        self.hop_size = hop_size

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def pad(self, pcm):
        n_samples = pcm.shape[0]
        pad_size = (self.hop_size - n_samples % self.hop_size) % self.hop_size
        pcm = np.pad(pcm, [0, pad_size], mode='constant')
        return pcm

    def __call__(self, mix, speech):
        mix = self.pad(mix)
        speech = self.pad(speech)

        if self.training:
            return mix, speech
        else:
            stoi = compute_stoi.stoi(speech, mix, 16000)
            pesq = pypesq.pesq(speech, mix, 16000)
            return mix, speech, {'stoi':stoi, 'pesq':pesq}

# def preprocess(feats, labels):
#     feats, lens = zip(
#         *[[torch.ShortTensor((f * 32767).astype(np.int16)), f.shape[0]]
#           for f in feats])
#     labels = [torch.ShortTensor((l * 32767).astype(np.int16)) for l in labels]
#     lens = torch.LongTensor(lens)
#     feats = pad_sequence(feats)
#     labels = pad_sequence(labels)

#     lens, ids = torch.sort(lens, descending=True)
#     feats = feats[ids]
#     labels = labels[ids]

#     return (feats, lens), (labels, lens), ids


# def pad_batch_train(batch, **kwargs):
#     feats, labels = zip(*batch)
#     feats, labels, _ = preprocess(feats, labels)
#     return feats, labels


# def pad_batch_test(batch, **kwargs):
#     feats, labels, extras = zip(*batch)
#     feats, labels, ids = preprocess(feats, labels)
#     extras = [extras[i] for i in ids]
#     return feats, labels, extras


# def collate_fn(batch, mode='train', **kwargs):
#     if mode == 'train':
#         return pad_batch_train(batch, **kwargs)
#     else:
#         return pad_batch_test(batch, **kwargs)

if __name__ == '__main__':
    import librosa
    from scipy.io import wavfile

    transform = Transform(training=False, hop_size=256)
    sr, pcm = wavfile.read('/home/zhaochengshuai/dataset/zheng_fan/FanData/HighPower_NormalSpeech/0001.cgmm.wav')
    mix, speech, metric = transform(pcm, pcm)
    print(mix.shape, speech.shape, metric)