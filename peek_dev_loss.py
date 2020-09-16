import os
import sys
import torch
import yaml
import soundfile
from functools import partial
import datetime
from tqdm import tqdm
sys.path.append(os.path.abspath('util'))
from dynamic_import import import_class
import zcs_print as zp
import zcs_visdom as zv
import save_load as sl
import average_meter as am

def get_dev_dataloader(conf):
    transform = import_class(conf['transform']['name'])(**conf['transform']['args'])
    transform.eval()
    collate_fn = import_class(conf['transform']['collate_fn']['name'])
    collate_fn = partial(collate_fn, mode='eval')
    dataset = import_class(conf['dataset']['dev']['name'])(transform=transform, **conf['dataset']['dev']['args'])
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=conf['eval']['dataloader']['batch_size'],
        shuffle=conf['train']['dataloader']['shuffle'],
        # sampler=,
        num_workers=conf['eval']['dataloader']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False)
    return dataloader

@torch.no_grad()
def validation(model, loss_fn, dataloader, vis, conf):
    model.eval()
    avg_loss = am.AverageMeter()
    datas = []
    for step, (batch_x, batch_y, extra) in tqdm(enumerate(dataloader)):
        # print(step, batch_x.shape, batch_y.shape)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        batch_pred, _ = model(batch_x)
        # print(batch_x.shape, pred.shape)
        # soundfile.write('./clean%d.wav' % step, data=batch_x.cpu().detach().numpy()[0], samplerate=16000)
        # soundfile.write('./pred%d.wav' % step, data=pred.cpu().detach().numpy()[0], samplerate=16000)
        loss = loss_fn(batch_pred, batch_y)
        avg_loss.update(loss.cpu().detach().numpy())

        for i, e in enumerate(extra):
            clean = e['clean']
            pred = batch_pred.cpu().detach().numpy()[i][:len(clean)]
            data = {'clean':clean, 'pred':pred, 'snr':e['snr'], 'metric':e['metric']} #处理后，会增加stoi和pesq两个字段，和metrix字段中的stoi和pesq做差值
            datas.append(data)

        # if step > 5:
        #     break

    evaluator = import_class(conf['eval']['evaluator']['name'])(datas, conf['eval']['evaluator']['num_workers'])
    avg_stoi = {}
    avg_pesq = {}
    for e in tqdm(evaluator):
        snr =  e['snr']
        if snr not in avg_stoi.keys():
            avg_stoi[snr] = am.AverageMeter()
            avg_pesq[snr] = am.AverageMeter()
        avg_stoi[snr].update(e['stoi'] - e['metric']['stoi'])
        avg_pesq[snr].update(e['pesq'] - e['metric']['pesq'])

    # print(avg_loss.avg, avg_stoi.avg, avg_pesq.avg)
    # vis.append([avg_loss.avg], 'dev_loss', opts={'title':'dev_loss', 'legend':['loss']})
    # vis.append([avg_stoi.avg, avg_pesq.avg], 'dev_metric', opts={'title':'dev_metric', 'legend':['stoi', 'pesq']})
    for snr in avg_stoi.keys():
        print('snr{} stoi: {} items, improve {} in average'.format(snr, avg_stoi[snr].count, avg_stoi[snr].avg))
        print('snr{} pesq: {} items, improve {} in average'.format(snr, avg_pesq[snr].count, avg_pesq[snr].avg))

def main(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)
    # print(conf)

    vis = zv.ZcsVisdom(server=conf['visdom']['ip'], port=conf['visdom']['port'])

    dev_dataloader = get_dev_dataloader(conf)

    model = import_class(conf['model']['name'])(**conf['model']['args'])
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu_ids']
    n_gpus = torch.cuda.device_count()
    zp.B('use %d gpus [%s]' % (n_gpus, conf['train']['gpu_ids']))

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    # model = sl.load_model(conf['checkpoint'], -1, model)

    loss_fn = import_class(conf['loss']['name'])(**conf['loss']['args'])
    loss_fn.cuda()

    for i in range(0, 40):
        model = sl.load_model(conf['checkpoint'], i, model)
        validation(model, loss_fn, dev_dataloader, vis, conf)
        break

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # main(sys.argv[1])
    main('./conf/nf_rnorm.yaml')
