import os
import sys
import torch
import yaml
from functools import partial
from tqdm import tqdm
sys.path.append(os.path.abspath('util'))
from dynamic_import import import_class
import zcs_print as zp
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

def train(model, loss_fn, optimizer, dataloader, epoch):
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

        # if step >= 50:
        #     break
    pbar.close()

def main(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)
    # print(conf)

    train_dataloader = get_train_dataloader(conf)

    model = import_class(conf['model']['name'])(**conf['model']['args'])
    print('total parameter:', model.total_parameter())
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu_ids']
    n_gpus = torch.cuda.device_count()
    zp.B('use %d gpus [%s]' % (n_gpus, conf['train']['gpu_ids']))

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    #model = sl.load_model(conf['checkpoint'], -1, model)

    loss_fn = import_class(conf['loss']['name'])(**conf['loss']['args'])
    loss_fn.cuda()

    optimizer = import_class(conf['train']['optimizer']['name'])(model.parameters(), **conf['train']['optimizer']['args'])
    #optimizer = sl.load_optimizer(conf['checkpoint'], -1, optimizer)

    zp.B('totally %d steps per epoch' % (len(train_dataloader)))
    trained_epoch = 1
    while trained_epoch < conf['train']['num_epochs']:
        #validation(model, loss_fn, dev_dataloader, vis, conf)
        train(model, loss_fn, optimizer, train_dataloader, trained_epoch)
        #sl.save_checkpoint(conf['checkpoint'], epoch, model, optimizer)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
