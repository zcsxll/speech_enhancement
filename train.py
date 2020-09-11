import os
import sys
import torch
import yaml
sys.path.append(os.path.abspath('util'))
from dynamic_import import import_class
import zcs_print as zp
import zcs_visdom as zv
import save_load as sl
import average_meter as am

def get_train_dataloader(conf):
    transform = import_class(conf['transform']['name'])(**conf['transform']['args'])
    collate_fn = import_class(conf['transform']['collate_fn']['name'])
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
    collate_fn = import_class(conf['transform']['collate_fn']['name'])
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

def train(model, loss_fn, optimizer, dataloader, vis):
    model.train()
    for step, (batch_x, batch_y) in enumerate(dataloader):
        # print(step, batch_x.shape, batch_y.shape)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pred, _ = model(batch_x)

        loss = loss_fn(pred, batch_y)
        print('%d/%d' % (step, len(dataloader)), loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            vis.append([loss.cpu().detach().numpy()], 'train_loss', opts={'title':'train_loss', 'legend':['loss']})
        # if step >= 50:
        #     break

@torch.no_grad()
def validation(model, loss_fn, dataloader, vis):
    model.eval()
    avegrage_meter = am.AverageMeter()
    for step, (batch_x, batch_y) in enumerate(dataloader):
        # print(step, batch_x.shape, batch_y.shape)

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pred, _ = model(batch_x)

        loss = loss_fn(pred, batch_y)
        print('%d/%d' % (step, len(dataloader)), loss.cpu().detach().numpy())
        avegrage_meter.update(loss.cpu().detach().numpy())

        # if step % 10 == 0:
        vis.append([loss.cpu().detach().numpy()], 'dev_loss', opts={'title':'dev_loss', 'legend':['loss']})
    vis.append([avegrage_meter.avg], 'avg_dev_loss', opts={'title':'avg_dev_loss', 'legend':['loss']})

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
    for epoch in range(trained_epoch + 1, conf['train']['num_epochs']):
        validation(model, loss_fn, dev_dataloader, vis)
        train(model, loss_fn, optimizer, train_dataloader, vis)
        sl.save_checkpoint(conf['checkpoint'], epoch, model, optimizer)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])