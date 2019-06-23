import sys

sys.argv = [
    __file__,
    "-a","resnet50",
    "--gpu", "0",
    '/mnt/local/aharonchiko' #imagenet directory
]
import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import gridspec

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='Augmentation.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    Accs = []
    Jaggednesses = []
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    archs = ['vgg16','resnet50','densenet201']
    jaggArchs = []
    for n in archs:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(n))
            model = models.__dict__[n](pretrained=True)
        else:
            print("=> creating model '{}'".format(n))
            model = models.__dict__[n]()




        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(args.workers / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if n.startswith('alexnet') or n.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion) and optimizer

        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


        # optionally resume from a checkpoint


        cudnn.benchmark = True

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.3, 0.8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None


        Sizes = np.arange(0.1,2.1,0.3)
        SizesLoaders = []
        for s in Sizes:
            if (s<=1.1):
                SizesLoaders.append(torch.utils.data.DataLoader(
                    datasets.ImageFolder(valdir, transforms.Compose([
                        transforms.RandomResizedCrop(234, scale=(s, s)),
                        transforms.ToTensor(),
                        normalize,
                    ])),
                    batch_size=1, shuffle=True,
                    num_workers=args.workers, pin_memory=True))
            else:
                SizesLoaders.append(torch.utils.data.DataLoader(
                    datasets.ImageFolder(valdir, transforms.Compose([
                        transforms.RandomResizedCenterCrop(int(224/s), scale=(1, 1)),
                        # transforms.ColorJitter(brightness=(1,1), contrast=(3.5,3.5), saturation=(1,1)),
                        transforms.ToTensor(),
                        normalize,
                    ])),
                    batch_size=1, shuffle=True,
                    num_workers=args.workers, pin_memory=True))


        jaggSizes = []
        for l in range(Sizes.shape[0]):
            jaggSizes.append(JaggedNess(SizesLoaders[l], model,args,Sizes[l]))

        jaggArchs.append([jaggSizes])

    #Plot results
    jaggArchs = np.array(jaggArchs)
    np.save('jaggArchs',jaggArchs)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['xtick.labelsize']=4
    plt.rcParams['ytick.labelsize']=4
    plt.close('all')
    # plt.figure(figsize=(10,10))
    ax = plt.subplot(311)
    ax.plot(Sizes,jaggArchs[0,0,:,0],'o-r',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[1,0,:,0],'o-g',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[2,0,:,0],'o-b',markersize=4,alpha=0.7)
    plt.xticks(Sizes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.xlabel('Crop size (of the original size)')
    plt.ylabel('P(failure)')
    plt.tight_layout()
    # plt.legend(archs)



    ax = plt.subplot(312)
    ax.plot(Sizes,jaggArchs[0,0,:,1],'o-r',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[1,0,:,1],'o-g',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[2,0,:,1],'o-b',markersize=4,alpha=0.7)
    plt.xticks(Sizes)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.xlabel('Crop size (of the original size)')
    plt.ylabel('Mean absolute change')
    plt.tight_layout()
    # plt.legend(archs)

    ax = plt.subplot(313)
    ax.plot(Sizes,jaggArchs[0,0,:,2],'o-r',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[1,0,:,2],'o-g',markersize=4,alpha=0.7)
    ax.plot(Sizes,jaggArchs[2,0,:,2],'o-b',markersize=4,alpha=0.7)
    plt.xticks(Sizes)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Crop size')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    # plt.legend(archs)
    # plt.show()
    plt.savefig('CheckTranslationInvariance.pdf',bbox_inches='tight')
    print('')


def JaggedNess(L, model,args,size):
    soft = nn.Softmax(dim=1)
    model.eval()
    Jagged = 0
    MAC = 0
    Counter = 0
    Accc = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(L):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if (int((input.shape[3] - 224 - 1)/2)>0): #Crop protocol
                center = int((input.shape[3] - 224 - 1)/2)
                for c in range(center-4,center+4):
                    inputs1 = input[:,:,c:c+224,c:c+224].cuda()
                    inputs2 = input[:,:,c+1:c+224+1,c+1:c+224+1].cuda()

                    target = target.cuda()


                    outputs1 = soft(model(inputs1))
                    outputs2 = soft(model(inputs2))
                    Accc += (torch.argmax(outputs1,1) == target).type('torch.DoubleTensor')
                    pred1 = outputs1[0,torch.argmax(outputs1,1).detach().cpu().numpy()].detach().cpu().numpy()
                    pred2 = outputs2[0,torch.argmax(outputs1,1).detach().cpu().numpy()].detach().cpu().numpy()
                    MAC += np.abs(pred1 - pred2) #Mean absolute change
                    Jagged += torch.sum(torch.argmax(outputs1,1) != torch.argmax(outputs2,1)).type('torch.DoubleTensor') #P(top-1 change)
                    Counter += float(inputs1.shape[0])
                    # print(Counter)
                    # print(Jagged / Counter)
                print(i)
                if (i == 1000):
                    print((Accc/Counter).detach().cpu().numpy())
                    return (Jagged / Counter).detach().cpu().numpy(),MAC / Counter,(Accc/Counter).detach().cpu().numpy()
            else: #Black background protocol
                center = int((224 - input.shape[3] - 1)/2)
                for c in range(center-4,center+4):
                    inputs1 = torch.zeros((1,3,224,224)).type('torch.FloatTensor').cuda()
                    inputs2 = torch.zeros((1,3,224,224)).type('torch.FloatTensor').cuda()

                    inputs1[:,:,c:c+input.shape[3],c:c+input.shape[3]] = input.cuda()
                    inputs2[:,:,c+1:c+input.shape[3]+1,c+1:c+input.shape[3]+1] = input.cuda()
                    # I = np.squeeze(inputs1.detach().cpu().numpy())
                    # I = np.transpose(I,(1,2,0))
                    # plt.close('all')
                    # plt.imshow(I)
                    # plt.show()
                    target = target.cuda()
                    # target[target == 0] = 155
                    # target[target == 1] = 156
                    outputs1 = soft(model(inputs1))
                    outputs2 = soft(model(inputs2))
                    Accc += (torch.argmax(outputs1,1) == target).type('torch.DoubleTensor')

                    pred1 = outputs1[0,torch.argmax(outputs1,1).detach().cpu().numpy()].detach().cpu().numpy()
                    pred2 = outputs2[0,torch.argmax(outputs1,1).detach().cpu().numpy()].detach().cpu().numpy()
                    MAC += np.abs(pred1 - pred2) #Mean absolute change
                    Jagged += torch.sum(torch.argmax(outputs1,1) != torch.argmax(outputs2,1)).type('torch.DoubleTensor') #P(top-1 change)
                    Counter += float(inputs1.shape[0])
                    # print(Counter)
                    # print(Jagged / Counter)
                print(i)
                if (i == 1000):
                    print((Accc/Counter).detach().cpu().numpy())
                    return (Jagged / Counter).detach().cpu().numpy(), MAC / Counter,(Accc/Counter).detach().cpu().numpy()




if __name__ == '__main__':
    main()