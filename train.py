import os
import sys
import torch
import yaml
from functools import partial
from tqdm import tqdm
sys.path.append(os.path.abspath('util'))
from dynamic_import import import_class
import zcs_print as zp
import zcs_visdom as zv
import save_load as sl
import average_meter as am

def get_train_dataloader(conf):
    transform = import_class(conf['transform']['name'])(**conf['transform']['args'])
    transform.train()
    collate_fn = import_class(conf['transform']['collate_fn']['name'])
    collate_fn = partial(collate_fn, mode='train')
    dataset = import_class(conf['dataset']['train']['name'])(transform=transform, **conf['dataset']['train']['args'])
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=conf['train']['dataloader']['batch_size'],
        shuffle=conf['train']['dataloader']['shuffle'],
        # sampler=,
        num_workers=conf['train']['dataloader']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True)
    return dataloader

def get_dev_dataloader(conf):
    transform = import_class(conf['transform']['name'])(**conf['transform']['args'])
    transform.eval()
    collate_fn = import_class(conf['transform']['collate_fn']['name'])
    collate_fn = partial(collate_fn, mode='eval')
    dataset = import_class(conf['dataset']['dev']['name'])(transform=transform, **conf['dataset']['dev']['args'])
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        # sampler=,
        num_workers=conf['train']['dataloader']['num_workers'], #和train保持一致
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False)
    return dataloader

def train(model, loss_fn, optimizer, dataloader, vis, epoch):
    model.train()
    pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    pbar.set_description(f'Epoch %d' % epoch)
    for step, (batch_x, batch_y) in enumerate(dataloader):
        # print(step, batch_x.shape, batch_y.shape)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pred, _ = model(batch_x)

        loss = loss_fn(pred, batch_y)
        # print('%d/%d' % (step, len(dataloader)), loss.cpu().detach().numpy())
        pbar.set_postfix(**{'sdr':loss.detach().cpu().item()})
        pbar.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            vis.append([loss.cpu().detach().numpy()], 'train_loss', opts={'title':'train_loss', 'legend':['loss']})
        # if step >= 50:
        #     break

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
        loss = loss_fn(batch_pred, batch_y)
        avg_loss.update(loss.cpu().detach().numpy())

        for i, e in enumerate(extra):
            clean = e['clean']
            pred = batch_pred.cpu().detach().numpy()[i][:len(clean)]
            data = {'clean':clean, 'pred':pred, 'snr':e['snr'], 'metric':e['metric']} #处理后，会增加stoi和pesq两个字段，和metrix字段中的stoi和pesq做差值
            datas.append(data)

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

    train_dataloader = get_train_dataloader(conf)
    dev_dataloader = get_dev_dataloader(conf)

    model = import_class(conf['model']['name'])(**conf['model']['args'])
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu_ids']
    n_gpus = torch.cuda.device_count()
    zp.B('use %d gpus [%s]' % (n_gpus, conf['train']['gpu_ids']))

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    model = sl.load_model(conf['checkpoint'], -1, model)

    loss_fn = import_class(conf['loss']['name'])(**conf['loss']['args'])
    loss_fn.cuda()

    optimizer = import_class(conf['train']['optimizer']['name'])(model.parameters(), **conf['train']['optimizer']['args'])
    optimizer = sl.load_optimizer(conf['checkpoint'], -1, optimizer)

    zp.B('totally %d steps per epoch' % (len(train_dataloader)))
    try:
        trained_epoch = sl.find_last_checkpoint(conf['checkpoint'])
        zp.B('train form epoch %d' % (trained_epoch + 1))
    except Exception as e:
        zp.B('train from the very begining, {}'.format(e))
        trained_epoch = -1
    # for epoch in range(trained_epoch + 1, conf['train']['num_epochs']):
    for epoch in range(0, conf['train']['num_epochs']):
        validation(model, loss_fn, dev_dataloader, vis, conf)
        train(model, loss_fn, optimizer, train_dataloader, vis, epoch)
        sl.save_checkpoint(conf['checkpoint'], epoch, model, optimizer)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])