import os
import sys
import numpy as np
import torch
import random
sys.path.append(os.path.join(os.path.abspath('.'), 'util'))
import pack_util
import data_util

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, **conf):
        self.transform = transform
        self.chunk_size = conf['chunk_size']

        self.sess_dict = {}
        self.sess_dict.update(pack_util.init_session(conf['clean']))
        self.length, self.clean_samples = pack_util.get_total(conf['clean'])

        self.snrs = []
        self.noise_prob = []
        for noise in conf['noise'].items():
            self.noise_prob += [noise[1]['prob']]
            self.snrs += [noise[1]['snr']]
        self.noise_prob = np.array(self.noise_prob)
        self.noise_prob = np.exp(self.noise_prob) / np.exp(self.noise_prob).sum()
        self.sess_dict.update(pack_util.init_session(conf['noise'].keys()))
        self.length, self.noise_samples = pack_util.get_total(conf['noise'].keys(), keep_struct=True)

    def __getitem__(self, idx):
        clean_sample = self.clean_samples[idx]
        clean = pack_util.load_audio(clean_sample, self.sess_dict, self.chunk_size)

        idx1 = np.where(np.random.multinomial(1, self.noise_prob) == 1)[0][0]
        idx2 = random.randint(0, len(self.noise_samples[idx1])-1)
        noise = pack_util.load_audio(self.noise_samples[idx1][idx2], self.sess_dict, self.chunk_size)

        snr = self.snrs[idx1]
        snr = snr[random.randint(0, len(snr)-1)]

        clean, noise, mix = data_util.gen_mix(clean, noise, snr, energy_norm=False)

        # rescale to [-1, 1]
        max_amp = max(np.max(np.abs(clean)), np.max(np.abs(mix))) + 1e-4
        scale = 1 / max_amp * 10**(random.randint(-30, 0) / 20)
        clean = clean * scale
        mix = mix * scale

        clean = np.where(np.isnan(mix), np.zeros_like(clean), clean)
        clean = np.where(np.isinf(mix), np.zeros_like(clean), clean)
        mix = np.where(np.isnan(mix), np.zeros_like(mix), mix)
        mix = np.where(np.isinf(mix), np.zeros_like(mix), mix)

        if self.transform is not None:
            return self.transform(mix, clean)
        return mix, clean

    def __len__(self):
        return self.length

if __name__ == '__main__':
    import yaml
    import soundfile

    with open('./conf/nf_rnorm.yaml') as fp:
        conf = yaml.safe_load(fp)

    dataset = Dataset(**conf['dataset']['train']['args'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)
    for step, (batch_x, batch_y) in enumerate(dataloader):
        print('step: %d' % (step), batch_x.shape, batch_y.shape)
        print(batch_x.detach().numpy().shape, type(batch_x.detach().numpy()[0][100]))
        soundfile.write(file='mix_%d.wav' % step, data=batch_x.detach().numpy()[0], samplerate=16000)
        soundfile.write(file='clean_%d.wav' % step, data=batch_y.detach().numpy()[0], samplerate=16000)
        if step == 2:
            break