import os
import sys
import torch
import numpy as np
import pypesq
sys.path.append(os.path.join(os.path.abspath('.'), 'util'))
import compute_stoi

class Transform():
    # def __init__(self, training=True, hop_size=256):
    def __init__(self, hop_size=256):
        # self.training = training
        self.hop_size = hop_size

    # def train(self):
    #     self.training = True

    # def eval(self):
    #     self.training = False

    def pad(self, pcm):
        n_samples = pcm.shape[0]
        pad_size = (self.hop_size - n_samples % self.hop_size) % self.hop_size
        pcm = np.pad(pcm, [0, pad_size], mode='constant')
        return pcm

    def __call__(self, mix, speech):
        mix = self.pad(mix)
        speech = self.pad(speech)

        # if self.training:
        #     return mix, speech
        # else:
        #     stoi = compute_stoi.stoi(speech, mix, 16000)
        #     pesq = pypesq.pesq(speech, mix, 16000)
        #     return mix, speech, {'stoi':stoi, 'pesq':pesq}
        return mix, speech

def preprocess(feats, labels):
    # feats, lens = zip(*[[torch.ShortTensor((f * 32767).astype(np.int16)), f.shape[0]] for f in feats])
    feats = [torch.ShortTensor((f * 32767).astype(np.int16)) for f in feats]
    labels = [torch.ShortTensor((l * 32767).astype(np.int16)) for l in labels]
    # lens = torch.LongTensor(lens)
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return feats, labels

    # lens, ids = torch.sort(lens, descending=True)
    # print(lens, ids)
    # feats = feats[ids]
    # labels = labels[ids]

    # return (feats, lens), (labels, lens), ids

def collate_fn(batch):
    feats, labels = zip(*batch)
    feats, labels = preprocess(feats, labels)
    return feats, labels

if __name__ == '__main__':
    import os
    import sys
    import yaml
    # import functools
    sys.path.append(os.path.abspath('dataset'))
    import near_field
    
    transform = Transform(hop_size=256)
    # collate_fn = functools.partial(collate_fn, mode='train')

    with open('./conf/nf_rnorm.yaml') as fp:
        conf = yaml.safe_load(fp)

    dataset = near_field.Dataset(transform=transform, **conf['dataset']['train']['args'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=collate_fn)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        print(step, batch_x.shape, batch_y.shape)
        if step >= 1:
            break

if __name__ == '__main__1':
    import librosa
    from scipy.io import wavfile

    transform = Transform(training=False, hop_size=256)
    sr, pcm = wavfile.read('/home/zhaochengshuai/dataset/zheng_fan/FanData/HighPower_NormalSpeech/0001.cgmm.wav')
    mix, speech, metric = transform(pcm, pcm)
    print(mix.shape, speech.shape, metric)