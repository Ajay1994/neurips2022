import os
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd


parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet20s', help='model architecture')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.9, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

best_sa = 0
args = parser.parse_args()
print(args)


# checkpoint = torch.load("0.0avg_gradient_checkpoint.pth.tar", map_location = torch.device('cuda:'+str(args.gpu)))
# baseline1 = checkpoint['result']['test_ta']

checkpoint = torch.load("runs/resnet50_cifar10_r90_seed1/0.9checkpoint.pth.tar", map_location = torch.device('cuda:'+str(args.gpu)))
baseline2 = checkpoint['result']['test_ta']
print("Baseline")
print(baseline2)
print("-"*50)
# checkpoint = torch.load("0.5avg_gradient_checkpoint.pth.tar", map_location = torch.device('cuda:'+str(args.gpu)))
# baseline3 = checkpoint['result']['test_ta']

# checkpoint = torch.load("0.75avg_gradient_checkpoint.pth.tar", map_location = torch.device('cuda:'+str(args.gpu)))
# baseline4 = checkpoint['result']['test_ta']
checkpoint = torch.load("runs/dense_resnet50_cifar10_r90_seed1/0.9checkpoint.pth.tar", map_location = torch.device('cuda:'+str(args.gpu)))
baseline4 = checkpoint['result']['test_ta']
print("Dense Assisted")
print(baseline4)

# plot comparison curve
plt.plot(np.arange(1, 181), baseline2 + [0] * (180 - len(baseline2)), label='Basline Training')
# plt.plot(np.arange(100, 182), baseline2[100:] , label='alpha = 0.25')
plt.plot(np.arange(1, 181), baseline4 + [0] * (180 - len(baseline4)), label='Dense Assisted Training')
# plt.plot(np.arange(100, 182), baseline4[100:] , label='Baseline')
# plt.plot(np.arange(1, 183), baseline4 + [0] * (183 - len(baseline4)), label='alpha = 0.75')
plt.legend()
plt.savefig("plots/comparison_resnet50_cifar10_r90.png")
plt.close()