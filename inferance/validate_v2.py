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
import cv2
import model_factory
from dataset import Dataset
from torchvision import utils as vutils


parser = argparse.ArgumentParser(description='PyTorch Validation Script on Validation dataset')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                    help='use multiple-gpus')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool for DPN models')

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
         Save tensor as picture
         :param input_tensor: tensor to save
         :param filename: saved file name
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
         # Make a copy
    input_tensor = input_tensor.clone().detach()
         # To cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
         # Denormalization
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

def main():
    args = parser.parse_args()

    test_time_pool = False
    if 'dpn' in args.model and args.img_size > 224 and not args.no_test_pool:
        test_time_pool = True

    if not args.checkpoint and not args.pretrained:
        args.pretrained = True 

    num_classes = 7
    batch_size = 1
    model = model_factory.create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        test_time_pool=test_time_pool)

    print('Model %s created, param count: %d' %
          (args.model, sum([m.numel() for m in model.parameters()])))

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.checkpoint))
    elif args.checkpoint:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
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

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            
            #save_image_tensor (images,"op.jpeg")

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    # Overall accuracy
    overall_accuracy=100 * correct / total
    print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,overall_accuracy))
    print("classes : ",class_names)

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')

    # Error rate 
    error_rate = (1 - (correct / total)) * 100
    print("Error Rate : ", error_rate)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('Per class accuracy')
    print('-'*18)
    i=0
    for label,accuracy in zip(eval_dataset.classes, class_accuracy):
        class_name=class_names[i]
        i+=1
        print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))




    
if __name__ == '__main__':
    main()


#python validate_v2.py --model resnet18 --checkpoint ./models/gps_lock_resnet18/model_best.pth.tar /media/ajithbalakrishnan/external/Dataset/freelancer/gps_lock_v2/test