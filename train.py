import os
import sys
import torch
import yaml
sys.path.append(os.path.abspath('util'))
from dynamic_import import import_class
import zcs_print as zp
import zcs_visdom as zv

def train(conf):
    with open(conf) as fp:
        conf = yaml.safe_load(fp)
    # print(conf)

    vis = zv.ZcsVisdom(server=conf['visdom']['ip'], port=conf['visdom']['port'])

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

    model = import_class(conf['model']['name'])(**conf['model']['args'])
    os.environ["CUDA_VISIBLE_DEVICES"] = conf['train']['gpu_ids']
    n_gpus = torch.cuda.device_count()
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    zp.B('use %d gpus [%s]' % (n_gpus, conf['train']['gpu_ids']))

    loss_fn = import_class(conf['loss']['name'])(**conf['loss']['args'])
    loss_fn.cuda()

    optimizer = import_class(conf['train']['optimizer']['name'])(model.parameters(), **conf['train']['optimizer']['args'])

    zp.B('totally %d steps per epoch' % (len(dataloader)))
    for epoch in range(conf['train']['num_epochs']):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            # print(step, batch_x.shape, batch_y.shape)

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            pred, _ = model(batch_x)

            loss = loss_fn(pred, batch_y)
            print(step, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                vis.append([loss.cpu().detach().numpy()], 'train_loss', opts={'title':'train_loss', 'legend':['loss']})
            if step % 1000 == 0:
                torch.save(model.state_dict(), './epoch_%03d_step_%05d.pth' % (epoch, step))
            # if step >= 20:
            #     break
        torch.save(model.state_dict(), './epoch_%03d.pth' % (epoch))

if __name__ == '__main__':
    assert len(sys.argv) == 2
    train(sys.argv[1])