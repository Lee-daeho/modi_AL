
# Python
import os
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
from datetime import datetime
from tqdm import tqdm
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from models.simsiam import SimSiam
from train_test import train, test
from load_dataset import load_dataset, load_sim_dataset
from selection_methods import query_samples
from config import *
from models.transformers import CONFIGS, VisionTransformer
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="TA-VAAL",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-b", "--base_model", type=str, default="resnet",
                    help="base model for train resnet for transformer")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "ViT-B_4"],
                    default="ViT-B_4",
                    help="Which variant to use.")
parser.add_argument("--pretrained_dir", type=str, default=None,
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--img_size", type=int, default=32,
                    help="Image size for training and test")
parser.add_argument("--self_supervised", action='store_true')
parser.add_argument("--sim_epoch", type=int, default=800,
                    help="SimSiam Network epochs")
parser.add_argument("--add_pretrained", type=str, default=None)
args = parser.parse_args()


def setup(args, NO_CLASSES):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = NO_CLASSES

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    #I do not use pretrained
    if args.pretrained_dir:
        model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)

    return args, model


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / SIM_EPOCH))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


##
# Main
if __name__ == '__main__':

    if not os.path.exists('sim_models'):
        os.mkdir('sim_models')
    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL']
    datasets = ['cifar10','cifar10im', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    assert args.base_model in ['resnet', 'transformer']
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    '''
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    time = datetime.now().strftime("%y-%m-%d-%H-%M")
    results = open('results_'+time + '_' +str(args.method_type)+"_"+args.dataset +'_' + args.base_model + '_self-supervised' + str(args.self_supervised)+ '.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        print('The entire datasize is {}'.format(len(data_train)))       
        ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        sim_model = None

        if args.self_supervised and not args.add_pretrained:
            SIM_EPOCH = args.sim_epoch

            init_lr = SIM_LR * SIM_BATCH / 256

            sim_model = SimSiam(resnet.ResNet18(zero_init_residual=True))

            sim_model.to(args.device)
            # sim_model = torch.nn.DataParallel(sim_model)

            sim_criterion = nn.CosineSimilarity(dim=1).cuda(args.device)

            optim_params = sim_model.parameters()
            sim_optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9)
            
            sim_train_dataset = load_sim_dataset(args.dataset)

            sim_train_loader = DataLoader(sim_train_dataset, batch_size=SIM_BATCH, shuffle=(True), pin_memory=True)

            losses = AverageMeter('Loss', ":.4f")

            for epoch in range(SIM_EPOCH):
                print('epoch : ',epoch)
                adjust_learning_rate(sim_optimizer, init_lr, epoch, args)

                #train
                sim_model.train()
                for i, (images, _) in tqdm(enumerate(sim_train_loader)):
                    images[0] = images[0].cuda(args.device)
                    images[1] = images[1].cuda(args.device)

                    p1, p2, z1, z2 = sim_model(x1=images[0], x2=images[1])
                    loss = -(sim_criterion(p1, z2).mean() + sim_criterion(p2, z1).mean()) * 0.5

                    losses.update(loss.item(), images[0].size(0))

                    sim_optimizer.zero_grad()
                    loss.backward()
                    sim_optimizer.step()
                
                if epoch % 10 == 0:
                    save_checkpoint({               
                    'epoch': epoch + 1,
                    'arch': args.base_model,
                    'state_dict': sim_model.state_dict(),
                    'optimizer' : sim_optimizer.state_dict(),
                }, is_best=False, filename='sim_models/' + 'checkpoint_{:04d}.pth.tar'.format(epoch))

        elif args.self_supervised and args.add_pretrained:
        
            sim_model = SimSiam(resnet.ResNet18(zero_init_residual=True))

            checkpoint = torch.load('sim_models/'+args.add_pretrained)

            print('loading pretrained weights {}'.format(args.add_pretrained))

            sim_model.load_state_dict(checkpoint['state_dict'])


        if args.total:
            labeled_set= indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True, drop_last=True)
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        for cycle in range(CYCLES):
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.base_model == 'resnet':
                    if args.dataset == "fashionmnist":
                        resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
                    else:
                        # For semi-supervised
                        if args.self_supervised:
                            resnet18 = sim_model.encoder.to(args.device)
                            # Reset model's fully connected layer
                            resnet18.fc = nn.Linear(512, NO_CLASSES).to(args.device)
                            
                            # Freeze Encoding part
                            for name, param in resnet18.named_parameters():
                                if name not in ['fc.weight', 'fc.bias']:
                                    param.requires_grad = False
                            # initiate fully connected layer
                            resnet18.fc.weight.data.normal_(mean=0.0, std=0.01)
                            resnet18.fc.bias.data.zero_()

                        else:
                            resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

                    if method == 'lloss' or method == 'TA-VAAL':
                        loss_module = LossNet(base_model = args.base_model).cuda()
                else:
                    args, transformer = setup(args, NO_CLASSES)

                    if method == 'lloss' or method == 'TA-VAAL':
                        loss_module = LossNet(num_channels = [CONFIGS[args.model_type].hidden_size]*4, base_model = args.base_model).cuda()

            if args.base_model == 'resnet':
                models      = {'backbone': resnet18}
                if method =='lloss' or method == 'TA-VAAL':
                    models = {'backbone': resnet18, 'module': loss_module}
                torch.backends.cudnn.benchmark = True
                
                # Loss, criterion and scheduler (re)initialization
                criterion      = nn.CrossEntropyLoss(reduction='none')
                if args.self_supervised:
                    # update only fully connected parameters
                    parameters = list(filter(lambda p: p.requires_grad, models['backbone'].parameters()))
                    LR=30
                else:
                    parameters = models['backbone'].parameters()
                    
                optim_backbone = optim.SGD(parameters, lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
    
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone}
                schedulers = {'backbone': sched_backbone}
                if method == 'lloss' or method == 'TA-VAAL':
                    optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WDECAY)
                    sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}
            else:
                ###Transformer
                models = {'backbone': transformer}
                criterion      = nn.CrossEntropyLoss(reduction='none')
                optim_backbone = torch.optim.SGD(models['backbone'].parameters(),
                            lr=LR_TR,
                            momentum=0.9,
                            weight_decay=WDECAY)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                torch.backends.cudnn.benchmark = True

                optimizers = {'backbone': optim_backbone}
                schedulers = {'backbone': sched_backbone}                

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            acc = test(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True)

    results.close()