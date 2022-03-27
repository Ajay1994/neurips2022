"""
    Create multiple prune masks and a random mask
    
    python -u train_mask.py \
    --data ../data \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 1 \
    --rate 0.9 \
    --save_dir ./runs/shared_seed1_resent18_cifar100_rate90
    python -u train_mask.py --data ../data --dataset cifar100 --arch resnet20s --seed 1 --rate 0.9 --save_dir ./runs/shared_seed1_resent20s_cifar100_rate90 
"""
import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

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

from utils import *
from pruner import *

parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
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
parser.add_argument('--rate', default=0.8, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

##################################### Gradient setting #################################################
parser.add_argument('--alpha', default=0.5, type=float, help='pruning rate')

best_sa = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4'
use_cuda = torch.cuda.is_available()

def main():
    global args, best_sa, fopen
    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    fopen = open(os.path.join(args.save_dir, 'base-mask.txt'), "a")
    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset and sparse and dense networks
    print('###################################### Creating Random Sparse Mask ######################################')
    model_sparse, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model_sparse.cuda()
    model_sparse.load_state_dict(torch.load(os.path.join(args.save_dir, 'init_weight.pt')))
    
    sparse_mask = torch.load(os.path.join(args.save_dir, 'mask.pt'))
    prune_model_custom(model_sparse, sparse_mask)
    check_sparsity(model_sparse)
    
    print("Sparse Model Loaded ...")
    
    criterion1 = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    
    optimizer1 = torch.optim.SGD(model_sparse.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=decreasing_lr, gamma=0.1)
    
    
    all_result = {}
    all_result['train_ta'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []

    start_epoch = 0
    start_state = 0
    
    print('######################################## Start Standard Training ########################################')
    for epoch in range(start_epoch, args.epochs):
        
        check_sparsity(model_sparse)
        if epoch > 1:
            print("\nEpoch : {} || LR1 = {:.3f} || Best Test Accuracy = {:.3f}".format(epoch, optimizer1.state_dict()['param_groups'][0]['lr'], max(all_result['test_ta'])))
        acc = train(train_loader, model_sparse, criterion1, optimizer1, epoch)
        
        # evaluate on validation set
        tacc = validate(val_loader, model_sparse, criterion1)
        # evaluate on test set
        test_tacc = validate(test_loader, model_sparse, criterion1)
        
        scheduler1.step()
        
        all_result['train_ta'].append(acc)
        all_result['val_ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)
        fopen.write("Epoch = {} \t||\t Train = {:.3f} % \t||\t Val = {:.3f} % \t||\t Test = {:.3f} %\n".format(epoch, acc, tacc, test_tacc))
        fopen.flush()
        

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc  > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint({
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model_sparse.state_dict(),
            'best_sa': best_sa,
            'optimizer': optimizer1.state_dict(),
            'scheduler': scheduler1.state_dict(),
        }, is_SA_best=is_best_sa, pruning=args.rate, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['train_ta'], label='train_acc')
        plt.plot(all_result['val_ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'base-mask.png'))
        plt.close()
        
    #report result
    print('######################################## Reporting Results ###########################################')
    check_sparsity(model_sparse)
    val_pick_best_epoch = np.argmax(np.array(all_result['val_ta']))
    print('Best SA = {}, Epoch = {}'.format(all_result['test_ta'][val_pick_best_epoch], val_pick_best_epoch+1))
    fopen.write('\nBest SA = {}\t\tEpoch = {}\n'.format(all_result['test_ta'][val_pick_best_epoch], val_pick_best_epoch+1))
#############################################################################################


def train(train_loader, model_sparse, criterion1, optimizer1, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model_sparse.train()

    start = time.time()
    for i, (image, target) in tqdm(enumerate(train_loader)):

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean_sparse = model_sparse(image)
        
        loss_sparse = criterion1(output_clean_sparse, target)
        
        optimizer1.zero_grad()
        loss_sparse.backward()
        optimizer1.step()

        output = output_clean_sparse.float()
        loss = loss_sparse.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    end = time.time()
    print('Time taken : {:.2f} sec \t||\t Train_accuracy {:.3f}'.format((end-start), top1.avg))

    return top1.avg

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    start = time.time()
    for i, (image, target) in tqdm(enumerate(val_loader)):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    end = time.time()
    print('Time taken : {:.2f} sec \t||\t Valid_accuracy {:.3f}'.format((end-start), top1.avg))

    return top1.avg


def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, "base-mask-"+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path,  "base-mask-model_SA_best.pth.tar"))

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    
if __name__ == '__main__':
    main()