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
        self.enorm = conf['enorm']
        dev_dir = os.path.expanduser(conf['dev_dir'])
        self.files = [os.path.join(dev_dir, i) for i in os.listdir(dev_dir)]
        self.files = [i for i in self.files if os.path.splitext(i)[-1] == '.npy']

    def __getitem__(self, idx):
        clean, mix = np.load(self.files[idx])
        if not self.enorm:
            clean = clean * (1 / 15)
            mix = mix * (1 / 15)
        
        filename = os.path.split(self.files[idx])[-1]
        _, noise_name, snr = filename.replace('.npy', '').split('_')
        
        extra = {}
        extra['name'] = filename
        extra['noise'] = noise_name
        extra['snr'] = int(snr)

        if self.transform is not None:
            return self.transform(mix, clean, extra)
        return mix, clean

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    import yaml
    import soundfile
    from dynamic_import import import_class
    sys.path.append(os.path.abspath('.'))

    with open('./conf/nf_rnorm.yaml') as fp:
        conf = yaml.safe_load(fp)

    transform = import_class(conf['transform']['name'])(**conf['transform']['args'])
    transform.eval()
    dataset = Dataset(transform=transform, **conf['dataset']['dev']['args'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)
    for step, (batch_x, batch_y, extra) in enumerate(dataloader):
        print('step: %d' % (step), batch_x.shape, batch_y.shape)
        print(extra.keys())
        # print(batch_x.detach().numpy().shape, type(batch_x.detach().numpy()[0][100]))
        # soundfile.write(file='mix_%d.wav' % step, data=batch_x.detach().numpy()[0], samplerate=16000)
        # soundfile.write(file='clean_%d.wav' % step, data=batch_y.detach().numpy()[0], samplerate=16000)
        if step == 2:
            break