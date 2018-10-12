from models import Model
from loss import Loss
from metric import Metric


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    add_arg('--batch_size', default=32, type=int, help='batch size')
    add_arg('--lr', default=1e-3, type=float, help='learning rate')
    add_arg('--num_workers', default=8, type=int, help='num of workers')
    add_arg('--gpu', default=True, help='using gpu')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    random.seed(1000)

    '''suggested format
    dataset = Dataset()
    dataloader = Dataloader()
    len_train = len(dataset) * 8 // 10
    len_val = len(dataset) - len_train
    train_dataloader, val_dataloader = dataloader.split([len_train, len_val])

    net = Net()
    if args.gpu:
        net = net.cuda()
        net = nn.DataParallel(net)
    
    criterion = Loss()
    metric = Metric()
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), 
                          lr=args.lr, 
                          momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = Model(net)
    model.compile(optimizer, criterion, metric, scheduler)
    model.fit(train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              epoch=args.epoch,
              use_gpu=args.gpu)
    '''