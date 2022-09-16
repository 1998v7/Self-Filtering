import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import argparse
from datetime import datetime
import mydataloader as dataloader
import torch.optim.lr_scheduler
import pandas as pd
from function import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--wdecay', default=5e-4, type=float, help='initial learning rate')

parser.add_argument('--noise_mode',  default='sym', choices=['sym', 'pair', 'instance'])
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--T', default=0.8, type=float, help='confidence threshold')
parser.add_argument('--k', default=2, type=int, help='queue length')

parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--semi', default='no', type=str)
parser.add_argument('--model', default='resnet18', type=str)

args = parser.parse_args()

if args.dataset == 'cifar10':
    args.data_path = './cifar-10'
    args.num_class = 10
elif args.dataset == 'cifar100':
    args.data_path = './cifar-100'
    args.num_class = 100
print(args)

set_env(args)

def build_model():
    if args.model == 'resnet32':
        from model.resnet32 import resnet32
        model = resnet32(args.num_class)
        print('============ use resnet32 ')
    elif args.model == 'resnet18':
        from model.resnet import ResNet18
        model = ResNet18(args.num_class)
        print('============ use resnet18 ')
    elif args.model == 'resnet34':
        from model.resnet import ResNet34
        model = ResNet34(args.num_class)
        print('============ use resnet34 ')
    model = model.cuda()
    return model

def main():
    test_log = open('./log/base/%s_%s_%.1f_k=%d' % (args.dataset, args.noise_mode, args.r, args.k) + '_test.txt', 'w')
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=8, root_dir=args.data_path, args=args,
                                         noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))
    net = build_model()
    memory_bank = []
    best_acc = 0.0
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    sch_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60], 0.1)

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    warmup_trainloader = loader.run('warmup')

    for epoch in range(args.num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        if epoch < args.warm_up:
            print('============ Warmup stage | lr = %.3f, T in penalty = %.3f' % (lr, args.T))
            _,  memory_bank = eval_train(net, memory_bank, eval_loader, args, epoch, test_log)
            warmup(epoch, net, optimizer, warmup_trainloader, args)
        else:
            print('============ Train stage | lr = %.3f, T in penalty = %.3f' % (lr, args.T))
            prob, pred, memory_bank = eval_train(net, memory_bank, eval_loader, args, epoch, test_log)
            labeled_trainloader = loader.run('train', pred, prob, test_log)
            train(epoch, net, optimizer, labeled_trainloader, args)

        test_acc = test(epoch, net, test_loader, test_log)
        print('\n')
        if test_acc > best_acc:
            best_acc = test_acc
        sch_lr.step()
    print('best test Acc: ', best_acc)
    test_log.write('Best Accuracy:%.2f\n' % (best_acc))


if __name__ == '__main__':
    main()