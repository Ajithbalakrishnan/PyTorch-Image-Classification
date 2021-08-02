from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import model_factory
from dataset import Dataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--restore-checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                    help='use multiple-gpus')
parser.add_argument('--no-test-pool', dest='test_time_pool', action='store_false',
                    help='use pre-trained model')


def main():
    args = parser.parse_args()

    # create model
    num_classes = 7
    model = model_factory.create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        test_time_pool=args.test_time_pool)

    # resume from a checkpoint
    if args.restore_checkpoint and os.path.isfile(args.restore_checkpoint):
        print("=> loading checkpoint '{}'".format(args.restore_checkpoint))
        checkpoint = torch.load(args.restore_checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.restore_checkpoint))
    elif not args.pretrained:
        print("=> no checkpoint found at '{}'".format(args.restore_checkpoint))
        exit(1)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True
    num_cpu = multiprocessing.cpu_count()

    eval_transform=transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    eval_dataset=datasets.ImageFolder(root=args.data, transform=eval_transform)
    eval_loader=data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes=len(eval_dataset.classes)
    dsize=len(eval_dataset)

    class_names=["baseballdiamond","forest","golfcourse","harbor","overpass","river","storagetanks"]

    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the test dataset
    correct = 0
    total = 0
    top5_ids = []
    batch_idx = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            print("images : ",images)
            images, labels = images.to(device), labels.to(device)
            batch_idx +=1
            outputs = model(images)

            top5 = outputs.topk(5)[1]
            top5_ids.append(top5.cpu().numpy())

        

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # if batch_idx % args.print_freq == 0:
            #     print('Predict: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            #         batch_idx, len(loader), batch_time=batch_time))

            # _, predicted = torch.max(outputs.data, 1)
            
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            # lbllist=torch.cat([lbllist,labels.view(-1).cpu()])


if __name__ == '__main__':
    main()


