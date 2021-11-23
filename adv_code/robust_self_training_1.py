"""
Main robust self-training script. Based loosely on code from
https://github.com/yaodongyu/TRADES
"""


import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter



import pandas as pd
import numpy as np

import time
import random

from utils import get_model

from losses import trades_loss, noise_loss
from datasets_new import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
from attack_pgd import pgd
from smoothing import quick_smoothing

from autoaugment import CIFAR10Policy
from cutout import Cutout

import logging


# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser(
    description='PyTorch TRADES Adversarial Training')

# Dataset config
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=DATASETS,
                    help='The dataset to use for training)')
parser.add_argument('--data_dir', default='data', type=str,
                    help='Directory where datasets are located')
parser.add_argument('--svhn_extra', action='store_true', default=False,
                    help='Adds the extra SVHN data')

# Model config
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='Name of the model (see utils.get_model)')
parser.add_argument('--model_dir', default='./rst-model',
                    help='Directory of model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Cancels the run if an appropriate checkpoint is found')
parser.add_argument('--normalize_input', action='store_true', default=False,
                    help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data'
                         ' fetching pipline)')

# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=5,
                    help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=25, type=int,
                    help='Checkpoint save frequency (in epochs)')

# Generic training configs
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed. '
                         'Note: fixing the random seed does not give complete '
                         'reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='Input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='Number of epochs to train. '
                         'Note: we arbitrarily define an epoch as a pass '
                         'through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training '
                         'configurations.')

# Eval config
parser.add_argument('--eval_freq', default=1, type=int,
                    help='Eval frequency (in epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int,
                    help='Maximum number for batches in training set eval')
parser.add_argument('--eval_attack_batches', default=1, type=int,
                    help='Number of eval batches to attack with PGD or certify '
                         'with randomized smoothing')

# Optimizer config
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=('trades', 'trades_fixed', 'cosine', 'wrn', 'trades_120_nis', 'trades_200', 'cyclic'),
                    help='Learning rate schedule')
parser.add_argument('--lr_min', default=0., type=float, help = 'for cyclic lr')
parser.add_argument('--lr_max', default=0.2, type=float, help = 'for cyclic lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='Use extragrdient steps')

# Adversarial / stability training config
parser.add_argument('--loss', default='trades', type=str,
                    choices=('trades', 'noise'),
                    help='Which loss to use: TRADES-like KL regularization '
                         'or noise augmentation')

parser.add_argument('--distance', '-d', default='l_2', type=str,
                    help='Metric for attack model: l_inf uses adversarial '
                         'training and l_2 uses stability training and '
                         'randomized smoothing certification',
                    choices=['l_inf', 'l_2'])
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='Adversarial perturbation size (takes the role of'
                         ' sigma for stability training)')

parser.add_argument('--pgd_num_steps', default=10, type=int,
                    help='number of pgd steps in adversarial training')
parser.add_argument('--pgd_step_size', default=0.007,
                    help='pgd steps size in adversarial training', type=float)
parser.add_argument('--beta', default=6.0, type=float,
                    help='stability regularization, i.e., 1/lambda in TRADES')

# Semi-supervised training configuration
parser.add_argument('--aux_data_filename', default=None, type=str,
                    help='Path to pickle file containing unlabeled data and '
                         'pseudo-labels used for RST')

parser.add_argument('--unsup_fraction', default=0.5, type=float,
                    help='Fraction of unlabeled examples in each batch; '
                         'implicitly sets the weight of unlabeled data in the '
                         'loss. If set to -1, batches are sampled from a '
                         'single pool')
parser.add_argument('--aux_take_amount', default=None, type=int,
                    help='Number of random aux examples to retain. '
                         'None retains all aux data.')

parser.add_argument('--remove_pseudo_labels', action='store_true',
                    default=False,
                    help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--entropy_weight', type=float,
                    default=0.0, help='Weight on entropy loss')

# Additional aggressive data augmentation
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='Use autoaugment for data augmentation')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='Use cutout for data augmentation')

args = parser.parse_args()

# ------------------------------ OUTPUT SETUP ----------------------------------
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Robust self-training')
logging.info('Args: %s', args)

#if not args.overwrite:
#    final_checkpoint_path = os.path.join(
#        model_dir, 'checkpoint-epoch{}.pt'.format(args.epochs))
#    if os.path.exists(final_checkpoint_path):
#        logging.info('Appropriate checkpoint found - quitting!')
#        sys.exit(0)
# ------------------------------------------------------------------------------

# ------------------------------- CUDA SETUP -----------------------------------
# should provide some improved performance
cudnn.benchmark = True
# useful setting for debugging
# cudnn.benchmark = False
# cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
# ------------------------------------------------------------------------------

# --------------------------- DATA AUGMENTATION --------------------------------
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
elif args.dataset == 'svhn':
    # the WRN paper does no augmentation on SVHN
    # obviously flipping is a bad idea, and it makes some sense not to
    # crop because there are a lot of distractor digits in the edges of the
    # image
    transform_train = transforms.ToTensor()

if args.autoaugment or args.cutout:
    assert (args.dataset == 'cifar10')
    transform_list = [
        transforms.RandomCrop(32, padding=4, fill=128),
        # fill parameter needs torchvision installed from source
        transforms.RandomHorizontalFlip()]
    if args.autoaugment:
        transform_list.append(CIFAR10Policy())
    transform_list.append(transforms.ToTensor())
    if args.cutout:
        transform_list.append(Cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(transform_list)
    logger.info('Applying aggressive training augmentation: %s'
                % transform_train)

transform_test = transforms.Compose([
    transforms.ToTensor()])
# ------------------------------------------------------------------------------

# ----------------- DATASET WITH AUX PSEUDO-LABELED DATA -----------------------
# transform_train = transforms.Compose([        
#         transforms.RandomCrop(size=32,padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),])

# transform_val = transforms.Compose([
#         transforms.ToTensor(),])

# transform_test = transforms.Compose([
#         transforms.ToTensor(),])

# train_set  = torchvision.datasets.CIFAR10(root='data', train=True , download=False, transform=transform_train )
# val_set    = torchvision.datasets.CIFAR10(root='data', train=True , download=False, transform=transform_val)
# test_set   = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
                          
# # Split training into train and validation
# train_size = 49000
# valid_size = 1000
# test_size  = 10000
# #get indices seed
# np.random.seed(0)
# indices    = np.arange(train_size+valid_size)
# np.random.shuffle(indices)
# train_indices = indices[0:train_size]
# val_indices   = indices[train_size:]
# #get data loader ofr train val and test
# #train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size ,sampler=SubsetRandomSampler(train_indices))
# val_loader   = DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=args.test_batch_size)
# test_loader  = DataLoader(test_set,batch_size=args.test_batch_size)

# trainset = SemiSupervisedDataset(base_dataset=args.dataset,
#                                  add_svhn_extra=args.svhn_extra,
#                                  take_amount = train_indices,
#                                  root=args.data_dir, train=True,
#                                  download=True, transform=transform_train,
#                                  aux_data_filename=args.aux_data_filename,
#                                  add_aux_labels=not args.remove_pseudo_labels,
#                                  aux_take_amount=args.aux_take_amount)

# # num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# # datapoints
# # TODO: make sure that this code works also when trainset.unsup_indices=[]
# train_batch_sampler = SemiSupervisedSampler(
#     trainset.sup_indices, trainset.unsup_indices,
#     args.batch_size, args.unsup_fraction,
#     num_batches=int(np.ceil(len(trainset.sup_indices) / args.batch_size)))
# epoch_size = len(train_batch_sampler) * args.batch_size

# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

# testset = SemiSupervisedDataset(base_dataset=args.dataset,
#                                 root=args.data_dir, train=False,
#                                 download=True,
#                                 transform=transform_test)
# test_loader = DataLoader(testset, batch_size=args.test_batch_size,
#                          shuffle=False, **kwargs)

# trainset_eval = SemiSupervisedDataset(
#     base_dataset=args.dataset,
#     add_svhn_extra=args.svhn_extra,
#     root=args.data_dir, train=True,
#     download=True, transform=transform_train)

# eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
#                                shuffle=True, **kwargs)

# eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
#                               shuffle=False, **kwargs)
###################################### Load data ###################################################
transform_train = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

transform_test = transforms.Compose([
        transforms.ToTensor(),])

train_set  = torchvision.datasets.CIFAR10(root='data', train=True , download=True, transform=transform_train)
val_set    = torchvision.datasets.CIFAR10(root='data', train=True , download=True, transform=transform_test)
test_set   = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)

# Split training into train and validation
train_size = 49000
valid_size = 1000
test_size  = 10000
epoch_size = train_size

# Applying balancing
# train_indices = list(range(50000))
# val_indices = []
# count = np.zeros(10)
# for index in range(len(train_set)):
#     _, target = train_set[index]
#     if(np.all(count==100)):
#         break
#     if(count[target]<100):
#         count[target] += 1
#         val_indices.append(index)
#         train_indices.remove(index)

# Unbalanced
np.random.seed(0)
indices    = np.arange(train_size+valid_size)
np.random.shuffle(indices)
train_indices = indices[0:train_size]
val_indices   = indices[train_size:]

print("Overlap indices:",list(set(train_indices) & set(val_indices)))
print("Size of train set:",len(train_indices))
print("Size of val set:",len(val_indices))
#get data loader ofr train val and test
#train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size ,sampler=SubsetRandomSampler(train_indices))
val_loader   = torch.utils.data.DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=args.test_batch_size)
test_loader   = torch.utils.data.DataLoader(test_set,batch_size=args.test_batch_size)
#TODO add new train_loader
trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 take_amount = train_indices,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)

# num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# datapoints
# TODO: make sure that this code works also when trainset.unsup_indices=[]
train_batch_sampler = SemiSupervisedSampler(
    trainset.sup_indices, trainset.unsup_indices,
    args.batch_size, args.unsup_fraction,
    num_batches=int(np.ceil(len(trainset.sup_indices) / args.batch_size)))
epoch_size = len(train_batch_sampler) * args.batch_size

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)
print('CIFAR10 dataloader: Done') 
# ------------------------------------------------------------------------------

# ----------------------- TRAIN AND EVAL FUNCTIONS -----------------------------
def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    train_metrics = []
    epsilon = args.epsilon
    #Calling dataloader here - Checking for item
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        #Inspecting item - item is tensor of size 64 with indices of data
        # with open('item_intrain.txt', 'a+') as infile:
        #     print('batch_idx is ', batch_idx, 'item is', item, file = infile)

        #TODO - Get indices of labelled data from item - tensor
        # lab_indices = (item <= train_size).nonzero().resize_((int(args.batch_size * (1 - args.unsup_fraction)),))
        # data_lab = data[lab_indices]
        # target_lab = target[lab_indices]
        # with open('lab_details.txt', 'a+') as infile:
        #     print('lab_indices are', lab_indices, file = infile)
        #     print('len of data_lab is', len(data_lab), 'len of target_lab is', len(target_lab), file = infile)

        # calculate robust loss
        if args.loss == 'trades':
            # The TRADES KL-robustness regularization term proposed by
            # Zhang et al., with some additional features
            (loss, natural_loss, robust_loss,
             entropy_loss_unlabeled) = trades_loss(
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                step_size=args.pgd_step_size,
                epsilon=epsilon,
                perturb_steps=args.pgd_num_steps,
                beta=args.beta,
                distance=args.distance,
                adversarial=args.distance == 'l_inf',
                entropy_weight=args.entropy_weight)

        elif args.loss == 'noise':
            # Augmenting the input with random noise as in Cohen et al.
            assert (args.distance == 'l_2')
            loss = noise_loss(model=model, x_natural=data,
                              y=target, clamp_x=True, epsilon=epsilon)
            entropy_loss_unlabeled = torch.Tensor([0.])
            natural_loss = robust_loss = loss

        loss.backward()
        optimizer.step()

        train_metrics.append(dict(
            epoch=epoch,
            loss=loss.item(),
            natural_loss=natural_loss.item(),
            robust_loss=robust_loss.item(),
            entropy_loss_unlabeled=entropy_loss_unlabeled.item()))

        # print progress
        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), epoch_size,
                           100. * batch_idx / len(train_loader), loss.item()))

        if args.lr_schedule == 'cyclic':
            scheduler.step()

    return train_metrics


def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx < args.eval_attack_batches:
                if args.distance == 'l_2':
                    # run coarse certification
                    incorrect_clean, incorrect_rob = quick_smoothing(
                        model, data, target,
                        sigma=args.epsilon,
                        eps=args.epsilon,
                        num_smooth=100, batch_size=1000)
                    pass
                elif args.distance == 'l_inf':
                    # run medium-strength gradient attack
                    is_correct_clean, is_correct_rob = pgd(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.pgd_num_steps,
                        step_size=args.pgd_step_size,
                        random_start=False)
                    incorrect_clean = (1-is_correct_clean).sum()
                    incorrect_rob = (1-np.prod(is_correct_rob, axis=1)).sum()
                else:
                    raise ValueError('No support for distance %s',
                                     args.distance)
                adv_correct_clean += (len(data) - int(incorrect_clean))
                adv_correct += (len(data) - int(incorrect_rob))
                adv_total += len(data)
            total += len(data)
            if ((eval_set == 'train') and
                    (batch_idx + 1 == args.train_eval_batches)):
                break
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    eval_data = dict(loss=loss, accuracy=accuracy,
                     robust_accuracy=robust_accuracy,
                     robust_clean_accuracy=robust_clean_accuracy)
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}
    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            'Smoothing' if args.distance == 'l_2' else 'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))

    return eval_data

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    schedule = args.lr_schedule
    # schedule from TRADES repo (different from paper due to bug there)
    if schedule == 'trades':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
    # schedule as in TRADES paper
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    # schedule as in WRN paper
    elif schedule == 'wrn':
        if epoch >= 0.3 * args.epochs:
            lr = args.lr * 0.2
        if epoch >= 0.6 * args.epochs:
            lr = args.lr * 0.2 * 0.2
        if epoch >= 0.8 * args.epochs:
            lr = args.lr * 0.2 * 0.2 * 0.2
    elif schedule == 'trades_120_nis':
        if epoch >= 75:
            lr = args.lr * 0.1
        if epoch >= 90:
            lr = args.lr * 0.01
        if epoch >= 100:
            lr = args.lr * 0.001
    elif schedule == 'trades_200':
        if epoch >= 125:
            lr = args.lr * 0.1
        if epoch >= 150:
            lr = args.lr * 0.01
        if epoch >= 170:
            lr = args.lr * 0.001
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# ------------------------------------------------------------------------------
#Supporting functions for new_eval
def FGSM_Attack_step(model,loss,image,target,eps=0.1,bounds=[0,1],GPU=0,steps=30): 
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps 
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img


def MSPGD(model,loss,data,target,eps=0.1,eps_iter=0.1,bounds=[],steps=[7,20,50]):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    img_arr = []
    for step in range(steps[-1]):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
        for j in range(len(steps)):
            if step == steps[j]-1:
                img_tmp = data + noise
                img_arr.append(img_tmp)
                break
    return img_arr

# ----------------------------- TRAINING LOOP ----------------------------------
def main():
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    num_classes = 10
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=args.normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # Handling cyclic schduler case
    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        logger.info('Using cyclic lr')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    else:
        scheduler = None
    #For training purpose.
    #loading from previous checkpoint
    # checkpoint = torch.load('.//rn18_adv_neweval_3/checkpoint-epoch18.pt')
    # model.load_state_dict(checkpoint['state_dict'])
    # checkpoint = torch.load('.//rn18_adv_neweval_3/opt-checkpoint_epoch18.tar')
    # optimizer.load_state_dict(checkpoint)                      
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        if args.lr_schedule == 'cyclic':
            lr = scheduler.get_lr()[0]
        else:
            lr = adjust_learning_rate(optimizer, epoch)

        logger.info('Setting learning rate to %g' % lr)
        # adversarial training
        train_data = train(args, model, device, train_loader, optimizer, scheduler, epoch)
        train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

        # save stats
        train_df.to_csv(os.path.join(model_dir, 'stats_train.csv'))
        eval_df.to_csv(os.path.join(model_dir, 'stats_eval.csv'))

        # save checkpoint
        #saving model for every epoch
        torch.save(dict(num_classes=num_classes,
                        state_dict=model.state_dict(),
                        normalize_input=args.normalize_input),
        os.path.join(model_dir,'checkpoint-epoch{}.pt'.format(epoch)))
        #saving optimizer only for save_freq
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir,
                                    'opt-checkpoint_epoch{}.tar'.format(epoch)))
        logging.info(str("Completed epoch" + str(epoch)))

        # evaluation on natural examples (not using here)
        #not evaluating here
        # logging.info(120 * '=')
        # if epoch % args.eval_freq == 0 or epoch == args.epochs:
        #     eval_data = {'epoch': int(epoch)}
        #     eval_data.update(
        #         eval(args, model, device, 'train', eval_train_loader))
        #     eval_data.update(
        #         eval(args, model, device, 'test', eval_test_loader))
        #     eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
        #     logging.info(120 * '=')

    
    #Code by nis for evaluation- starts here
    model.eval()
    loss = nn.CrossEntropyLoss()
    tensorboard_dir = os.path.join(args.model_dir, "runs")
    try:
        os.mkdir(tensorboard_dir)
    except:
        pass
    writer = SummaryWriter(log_dir = tensorboard_dir)


    ######################### FIND BEST MODEL###############################################
    EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log.txt') 
    ACC_EPOCH_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_acc_epoch.txt')
    ACC_IFGSM_EPOCH_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_ifgsm_acc_epoch.txt')
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = '##################### iter.FGSM: steps=7,eps=8.0/255,1####################\n'
    log_file.write(msg)
    log_file.close()
    VAL_CSV = os.path.join(args.model_dir, 'val_ifgsm.csv')
    val_file = open(VAL_CSV, 'w')
    val_file.write("epochs,val accuracy\n")
    val_file.close()
    accuracy_log = np.zeros(args.epochs+1)
    for epoch in range(1, args.epochs+1):
        model_name = os.path.join(args.model_dir, 'checkpoint-epoch'+str(epoch)+'.pt')
        model.load_state_dict(torch.load(model_name)['state_dict'])
        eps=8.0/255
        accuracy = 0
        accuracy_ifgsm = 0
        i = 0
        for data, target in val_loader:
            #data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=7)
            data   = Variable(data).cuda()
            target = Variable(target).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy = accuracy + prediction.eq(target.data).sum()
            i = i + 1 
        for data, target in val_loader:
            data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=7)
            data   = Variable(data).cuda()
            target = Variable(target).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy_ifgsm = accuracy_ifgsm + prediction.eq(target.data).sum()
        acc = (accuracy.item()*1.0) / (i*args.test_batch_size) * 100
        acc_ifgsm = (accuracy_ifgsm.item()*1.0) / (i*args.test_batch_size) * 100
        #log accuracy to file
        msg= str(epoch)+','+str(acc)+'\n'
        log_file = open(ACC_EPOCH_LOG_NAME,'a+')
        writer.add_scalar("Accuracy/Valid Acc", acc, epoch)
        log_file.write(msg)
        log_file.close()
        
        msg1= str(epoch)+','+str(acc_ifgsm)+'\n'
        log_file = open(ACC_IFGSM_EPOCH_LOG_NAME,'a+')
        log_file.write(msg1)
        log_file.close()

        val_file = open(VAL_CSV, 'a+')
        val_file.write(str(epoch)+','+str(acc_ifgsm)+'\n')
        val_file.close()

        accuracy_log[epoch] = acc_ifgsm
        writer.add_scalar("Accuracy/IFGSM-7 Acc", acc_ifgsm, epoch)
        #writer.export_scalars_to_json("./all_scalars.json")
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] : Acc:%f \t\t'
                %(epoch, args.epochs,acc))
        sys.stdout.flush() 

    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'IFGSM-7 on val set\n'
    log_file.write(msg)
    msg = 'Best Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
    log_file.write(msg)
    log_file.close()

    writer.add_text("Best Model",'Epoch:'+str(accuracy_log.argmax())+',  Acc:'+str(accuracy_log.max()),str(accuracy_log.argmax()))
    writer.close()

    model_name = os.path.join(args.model_dir, 'checkpoint-epoch' + str(accuracy_log.argmax())+'.pt')
    model.load_state_dict(torch.load(model_name)['state_dict'])
    best_model_dir = os.path.join(args.model_dir, 'best_models')
    try:
        os.mkdir(best_model_dir)
    except:
        pass
    best_model_name = os.path.join(best_model_dir, 'epoch' + str(accuracy_log.argmax())+'.pkl')
    torch.save(model.state_dict(),best_model_name)
    model.eval()
    model.cuda()
    # ##################################### FGSM #############################################
    EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log.txt')
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = '##################### FGSM ####################\n'
    log_file.write(msg)
    log_file.close()
    for eps in np.arange(0.0/255,10.0/255,2.0/255):
        i = 0
        accuracy = 0
        for data, target in test_loader:
            adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=1)
            data   = Variable(adv).cuda()
            target = Variable(target).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy = accuracy + prediction.eq(target.data).sum()
            i = i + 1
        acc = (accuracy.item()*1.0) / (test_size) * 100
        log_file = open(EVAL_LOG_NAME,'a+')
        msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
        log_file.write(msg)
        log_file.close()
    ##################################### iFGSM #############################################
    #Test set for chosen epoch
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = '##################### iFGSM: step=7 ####################\n'
    log_file.write(msg)
    log_file.close()
    for eps in np.arange(2.0/255,10.0/255,2.0/255):
        i = 0
        accuracy = 0
        for data, target in test_loader:
            adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=7)
            data   = Variable(adv).cuda()
            target = Variable(target).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy = accuracy + prediction.eq(target.data).sum()
            i = i + 1
        acc = (accuracy.item()*1.0) / (test_size) * 100
        log_file = open(EVAL_LOG_NAME,'a+')
        msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
        log_file.write(msg)
        log_file.close()


    ##################################### PGD, steps=[7,20,50,100,500] #############################################
    #On testset for chosen epoch
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = '##################### PGD: steps=[7,20,50,100],eps_iter=2/255 ####################\n'
    log_file.write(msg)
    log_file.close()
    all_steps = [7,20,50,100]
    num_steps = len(all_steps)
    eps = 8.0/255
    i = 0
    acc_arr = torch.zeros((num_steps))
    for data, target in test_loader:
        adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
        target = Variable(target).cuda()
        for j in range(num_steps):
            data   = Variable(adv_arr[j]).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()
        i = i + 1
    print(acc_arr)
    for j in range(num_steps):
        acc_arr[j] = (acc_arr[j].item()*1.0) / (test_size) * 100
        log_file = open(EVAL_LOG_NAME,'a+')
        msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
        log_file.write(msg)
        log_file.close()
        

    ##################################### iFGSM -20 for all epochs #############################################
    #VAL Set on all epochs
    EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_ifgsm20_val.txt')
    log_file = open(EVAL_LOG_NAME,'w')
    msg = '##################### iFGSM: step=20 ####################\n'
    log_file.write(msg)
    log_file.close()
    VAL_CSV = os.path.join(args.model_dir, 'val_ifgsm20.csv')
    val_file = open(VAL_CSV, 'w')
    val_file.write("epochs,8.0/255\n")
    val_file.close()
    accuracy_log = np.zeros(args.epochs+1)
    for epoch in range(1, args.epochs + 1):
        model_name = os.path.join(args.model_dir, 'checkpoint-epoch'+str(epoch)+'.pt')
        model.load_state_dict(torch.load(model_name)['state_dict'])
        eps = 8.0/255
        accuracy = 0
        for data, target in val_loader:
            adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=20)
            data   = Variable(adv).cuda()
            target = Variable(target).cuda()
            out = model(data)
            prediction = out.data.max(1)[1] 
            accuracy = accuracy + prediction.eq(target.data).sum()
        acc = (accuracy.item()*1.0) / (valid_size) * 100
        accuracy_log[epoch] = acc
        log_file = open(EVAL_LOG_NAME,'a+')
        msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
        log_file.write(msg)
        log_file.close()
        val_file = open(VAL_CSV, 'a+')
        val_file.write(str(epoch)+','+str(acc)+'\n')
        val_file.close()

    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'IFGSM-20 on val set\n'
    log_file.write(msg)
    msg = 'Best Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
    log_file.write(msg)
    log_file.close()

    # Commented from first - Uncomment all lines before this
    # ##################################### PGD, steps=50 #############################################
    # log_file = open(EVAL_LOG_NAME,'a+')
    # msg = '##################### PGD: steps=50,eps_iter=2/255 ####################\n'
    # log_file.write(msg)
    # log_file.close()
    # for eps in np.arange(2.0/255,10.0/255,2.0/255):
    #     i = 0
    #     accuracy = 0
    #     for data, target in test_loader:
    #         adv = PGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=50)
    #         data   = Variable(adv).cuda()
    #         target = Variable(target).cuda()
    #         out = model(data)
    #         prediction = out.data.max(1)[1] 
    #         accuracy = accuracy + prediction.eq(target.data).sum()
    #         i = i + 1
    #     acc = (accuracy.item()*1.0) / (test_size) * 100
    #     log_file = open(EVAL_LOG_NAME,'a+')
    #     msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
    #     log_file.write(msg)
    #     log_file.close()

    # # ##################################### PGD, steps=[20] for all epochs#############################################
    #VAL SET
    print('Running validation on val set')
    EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_valpgd.txt')
    log_file = open(EVAL_LOG_NAME,'a+')
    msg =  '##################### PGD on valset steps=[20],eps_iter=8/255 ####################\n'
    log_file.write(msg)
    log_file.close()
    TEST_CSV = os.path.join(args.model_dir, 'val_pgd20.csv')
    test_file = open(TEST_CSV, 'w')
    test_file.write("epochs,val robust accuracy,val clean accuracy\n")
    test_file.close()
    #all_steps = [7,20,50,100]
    all_steps = [20]
    num_steps = len(all_steps)
    eps = 8.0/255
    accuracy_log = np.zeros(args.epochs+1)
    for epoch in range(1, args.epochs+1):
        model_name = os.path.join(args.model_dir, 'checkpoint-epoch'+str(epoch)+'.pt')
        model.load_state_dict(torch.load(model_name)['state_dict'])
        i = 0
        acc_arr = torch.zeros((num_steps))
        acc_clean = torch.zeros((1))
        for data, target in val_loader:
            adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
            target = Variable(target).cuda()
            #For clean accuracy
            data_clean = Variable(data).cuda()
            out_clean = model(data_clean)
            prediction_clean = out_clean.data.max(1)[1]
            acc_clean[0] = acc_clean[0] + prediction_clean.eq(target.data).sum()
            #For robust accuracy
            for j in range(num_steps):
                data   = Variable(adv_arr[j]).cuda()
                out = model(data)
                prediction = out.data.max(1)[1] 
                acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()
            i = i + 1

        acc_clean[0] = (acc_clean[0].item()*1.0) / (valid_size) * 100
        print(acc_arr)
        for j in range(num_steps):
            acc_arr[j] = (acc_arr[j].item()*1.0) / (valid_size) * 100
            accuracy_log[epoch] = acc_arr[j]
            log_file = open(EVAL_LOG_NAME,'a+')
            #msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
            msg = 'epoch,' + str(epoch) + ' , Robust Acc,' + str(acc_arr[j]) + ' , Clean Acc,' + str(acc_clean[0]) + '\n'
            log_file.write(msg)
            log_file.close()
            test_file = open(TEST_CSV, 'a+')
            test_file.write(str(epoch)+','+str(acc_arr[j].item())+','+str(acc_clean[0].item())+'\n')
            test_file.close()

    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'PGD-20 on val set\n'
    log_file.write(msg)
    msg = 'Best Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
    log_file.write(msg)
    log_file.close()

    ##################################### PGD, steps=[20] for all epochs #############################################
    #TEST SET
    EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_testpgd.txt')
    log_file = open(EVAL_LOG_NAME,'a+')
    #msg = '##################### PGD: steps=[7,20,50,100],eps_iter=2/255 ####################\n'
    msg =  '##################### PGD on testset steps=[20],eps_iter=8/255 ####################\n'
    log_file.write(msg)
    log_file.close()
    TEST_CSV = os.path.join(args.model_dir, 'test_pgd20.csv')
    test_file = open(TEST_CSV, 'w')
    test_file.write("epochs,test accuracy\n")
    test_file.close()
    #all_steps = [7,20,50,100]
    all_steps = [20]
    num_steps = len(all_steps)
    eps = 8.0/255
    accuracy_log = np.zeros(args.epochs+1)
    for epoch in range(1, args.epochs+1):
        model_name = os.path.join(args.model_dir, 'checkpoint-epoch'+str(epoch)+'.pt')
        model.load_state_dict(torch.load(model_name)['state_dict'])
        i = 0
        acc_arr = torch.zeros((num_steps))
        for data, target in test_loader:
            adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
            target = Variable(target).cuda()
            for j in range(num_steps):
                data   = Variable(adv_arr[j]).cuda()
                out = model(data)
                prediction = out.data.max(1)[1] 
                acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()
            i = i + 1
        print(acc_arr)
        for j in range(num_steps):
            acc_arr[j] = (acc_arr[j].item()*1.0) / (test_size) * 100
            accuracy_log[epoch] = acc_arr[j]
            log_file = open(EVAL_LOG_NAME,'a+')
            #msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
            msg = 'epoch,' + str(epoch) + ' , Acc,' + str(acc_arr[j]) + '\n'
            log_file.write(msg)
            log_file.close()
            test_file = open(TEST_CSV, 'a+')
            test_file.write(str(epoch)+','+str(acc_arr[j].item())+'\n')
            test_file.close()
    log_file = open(EVAL_LOG_NAME,'a+')
    msg = 'PGD-20 on test set\n'
    log_file.write(msg)
    msg = 'Best Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
    log_file.write(msg)
    log_file.close()
    # ------------------------------------------------------------------------------
    ##################################### PGD, steps=[20] for all epochs#############################################
    #Train SET
    # EVAL_LOG_NAME = os.path.join(args.model_dir, 'evaluation_log_trainpgd.txt')
    # log_file = open(EVAL_LOG_NAME,'a+')
    # #msg = '##################### PGD: steps=[7,20,50,100],eps_iter=2/255 ####################\n'
    # msg =  '##################### PGD on valset steps=[20],eps_iter=8/255 ####################\n'
    # log_file.write(msg)
    # log_file.close()
    # TEST_CSV = os.path.join(args.model_dir, 'train_pgd20.csv')
    # test_file = open(TEST_CSV, 'w')
    # test_file.write("epochs,train accuracy\n")
    # test_file.close()
    # #all_steps = [7,20,50,100]
    # all_steps = [20]
    # num_steps = len(all_steps)
    # eps = 8.0/255
    # accuracy_log = np.zeros(args.epochs+1)
    # for epoch in range(1, args.epochs+1):
    #     model_name = os.path.join(args.model_dir, 'checkpoint-epoch'+str(epoch)+'.pt')
    #     model.load_state_dict(torch.load(model_name)['state_dict'])
    #     i = 0
    #     acc_arr = torch.zeros((num_steps))
    #     for batch_idx, ((data, target), item) in enumerate(train_loader):
    #         adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
    #         target = Variable(target).cuda()
    #         for j in range(num_steps):
    #             data   = Variable(adv_arr[j]).cuda()
    #             out = model(data)
    #             prediction = out.data.max(1)[1] 
    #             acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum()
    #         i = i + 1

    #     print(acc_arr)
    #     for j in range(num_steps):
    #         acc_arr[j] = (acc_arr[j].item()*1.0) / (train_size) * 100
    #         accuracy_log[epoch] = acc_arr[j]
    #         log_file = open(EVAL_LOG_NAME,'a+')
    #         #msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
    #         msg = 'epoch,' + str(epoch) + ' , Acc,' + str(acc_arr[j]) + '\n'
    #         log_file.write(msg)
    #         log_file.close()
    #         test_file = open(TEST_CSV, 'a+')
    #         test_file.write(str(epoch)+','+str(acc_arr[j].item())+'\n')
    #         test_file.close()

    # log_file = open(EVAL_LOG_NAME,'a+')
    # msg = 'PGD-20 on train set\n'
    # log_file.write(msg)
    # msg = 'Best Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
    # log_file.write(msg)
    # log_file.close()

if __name__ == '__main__':
    main()
