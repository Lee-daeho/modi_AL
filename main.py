
# Python
import os
import random
import math
import copy
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
import pickle
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from models.simsiam import SimSiam, Loss_SimSiam
from models.autoencoder import AutoEncoder
from train_test import train, test, test_tsne
from load_dataset import load_dataset, load_sim_dataset
from selection_methods import query_samples, get_kcg
from data.sampler import SubsetSequentialSampler
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
parser.add_argument("--add_llosspretrained", type=str, default="cifar10_lloss_checkpiont_0799.pth.tar")
parser.add_argument("--frozen", action='store_true')
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--initial", action="store_true")
parser.add_argument("--lloss", action="store_true")
parser.add_argument("--self_method", type=str, default="SimSiam")
parser.add_argument("--addednum", type=int, default=None)
parser.add_argument("--tsne", action='store_true')
parser.add_argument("--layer", type=int, default=-1)
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
    self_method = args.self_method
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL']
    datasets = ['cifar10','cifar10im', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    assert args.base_model in ['resnet', 'transformer']
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    '''
    if args.lr:
        LR = args.lr
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    time = datetime.now().strftime("%d-%H-%M")    
    foldername = 'results_'+time + '_' +str(args.method_type)+"_"+args.dataset + '_ssl_' + str(args.self_supervised)[0] +'_initial_'+ str(args.initial)[0] + '_lr_' + str(args.lr) + '_frozen_' + str(args.frozen)[0] + '_add_' + str(args.addednum) + '_layers_' + str(args.layer)
    
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    results = open(foldername + '/' + 'results_'+time + '_' +str(args.method_type)+"_"+args.dataset +'_' + '_ssl_' + str(args.self_supervised)[0]+ 
            '_initial_'+ str(args.initial)[0] + '_lr_' + str(args.lr) + '_frozen_' + str(args.frozen)[0] + '_add_' + str(args.addednum) + '.txt','w')
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
        if not args.addednum: 
            ADDENDUM = adden
        else:
            ADDENDUM = args.addednum
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        sim_model = None
        initial_data = None
        saving = None
        
        loss_model = None
        lloss_name = None

        if args.self_supervised:
            if not args.add_pretrained and self_method == 'SimSiam':
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
                if method == 'lloss' or method == 'TA-VAAL' and args.lloss:
                    loss_model = LossNet(base_model = args.base_model).cuda()
                    sim_lloss = Loss_SimSiam(loss_model)

                    sim_lloss.to(args.device)

                    lloss_criterion = nn.CosineSimilarity(dim=1).cuda(args.device)
                    lloss_optim_params = sim_lloss.parameters()
                    lloss_optimizer = torch.optim.SGD(lloss_optim_params, init_lr, momentum=0.9)

                losses = AverageMeter('Loss', ":.4f")

                for epoch in range(SIM_EPOCH):
                    print('epoch : ',epoch)
                    adjust_learning_rate(sim_optimizer, init_lr, epoch, args)

                    #train
                    sim_model.train()
                    for i, (images, _) in tqdm(enumerate(sim_train_loader)):
                        images[0] = images[0].cuda(args.device)
                        images[1] = images[1].cuda(args.device)

                        p1, p2, z1, z2, features1, features2 = sim_model(x1=images[0], x2=images[1])
                        loss = -(sim_criterion(p1, z2).mean() + sim_criterion(p2, z1).mean()) * 0.5

                        if method == 'lloss' or method == 'TA-VAAL' and args.lloss:
                            features1[0] = features1[0].detach()
                            features1[1] = features1[1].detach()
                            features1[2] = features1[2].detach()
                            features1[3] = features1[3].detach()

                            features2[0] = features2[0].detach()
                            features2[1] = features2[1].detach()
                            features2[2] = features2[2].detach()
                            features2[3] = features2[3].detach()

                            pl1, pl2, zl1, zl2 = sim_lloss(features1, features2)
                            lloss_loss = -(lloss_criterion(pl1, zl2).mean() + lloss_criterion(pl2, zl2).mean()) * 0.5

                        losses.update(loss.item(), images[0].size(0))

                        sim_optimizer.zero_grad()
                        loss.backward()
                        sim_optimizer.step()

                        if method == 'lloss' or method == 'TA-VAAL' and args.lloss:
                            lloss_optimizer.zero_grad()
                            lloss_loss.backward()
                            lloss_optimizer.step()
                    
                    if (epoch + 1) % 200 == 0:
                        args.add_pretrained = args.dataset + '_checkpoint_{:04d}.pth.tar'.format(epoch)
                        save_checkpoint({               
                        'epoch': epoch + 1,
                        'arch': args.base_model,
                        'state_dict': sim_model.state_dict(),
                        'optimizer' : sim_optimizer.state_dict(),
                    }, is_best=False, filename='sim_models/' + args.dataset +'_checkpoint_{:04d}.pth.tar'.format(epoch))

                        if args.lloss:
                            lloss_name = args.dataset +'_lloss_checkpoint_{:04d}.pth.tar'.format(epoch)
                            save_checkpoint({               
                            'epoch': epoch + 1,
                            'arch': args.base_model,
                            'state_dict': sim_lloss.state_dict(),
                            'optimizer' : lloss_optimizer.state_dict(),
                            }, is_best=False, filename='sim_models/' + args.dataset +'_lloss_checkpoint_{:04d}.pth.tar'.format(epoch))

            elif not args.add_pretrained and self_method == 'encoder':
                SIM_EPOCH = args.sim_epoch

                init_lr = SIM_LR * SIM_BATCH / 256

                auto_model = AutoEncoder(resnet.ResNet18(zero_init_residual=True))

                auto_model.to(args.device)
                                
                auto_criterion = nn.MSELoss().to(args.device)

                optim_params = auto_model.parameters()
                auto_optimizer = torch.optim.SGD(optim_params, init_lr, momentum=0.9)
                
                auto_train_dataset, _, _, _, _, _ = load_dataset(args.dataset)

                auto_train_loader = DataLoader(auto_train_dataset, batch_size=SIM_BATCH, shuffle=(True), pin_memory=True)
                if method == 'lloss' or method == 'TA-VAAL' and args.lloss:
                    loss_model = LossNet(base_model = args.base_model).cuda()
                    sim_lloss = Loss_SimSiam(loss_model)

                    sim_lloss.to(args.device)

                    lloss_criterion = nn.CosineSimilarity(dim=1).cuda(args.device)
                    lloss_optim_params = sim_lloss.parameters()
                    lloss_optimizer = torch.optim.SGD(lloss_optim_params, init_lr, momentum=0.9)

                losses = AverageMeter('Loss', ":.4f")

                for epoch in range(SIM_EPOCH):
                    print('epoch : ',epoch)
                    adjust_learning_rate(auto_optimizer, init_lr, epoch, args)

                    #train
                    auto_model.train()
                    for i, (image, _) in tqdm(enumerate(auto_train_loader)):
                        image = image.cuda(args.device)

                        pred = auto_model(image)
                        loss = auto_criterion(image, pred)

                        losses.update(loss.item(), image.size(0))

                        auto_optimizer.zero_grad()
                        loss.backward()
                        auto_optimizer.step()
                    
                    if (epoch + 1) % 200 == 0:
                        args.add_pretrained = self_method + '_' + args.dataset + '_checkpoint_{:04d}.pth.tar'.format(epoch)
                        save_checkpoint({               
                        'epoch': epoch + 1,
                        'arch': args.base_model,
                        'state_dict': auto_model.state_dict(),
                        'optimizer' : auto_optimizer.state_dict(),
                    }, is_best=False, filename='sim_models/' + self_method + '_' + args.dataset + '_checkpoint_{:04d}.pth.tar'.format(epoch))

            elif args.add_pretrained and self_method == 'SimSiam':
            
                sim_model = SimSiam(resnet.ResNet18(zero_init_residual=True))

                checkpoint = torch.load('sim_models/'+args.add_pretrained)

                print('loading pretrained weights {}'.format(args.add_pretrained))

                sim_model.load_state_dict(checkpoint['state_dict'])

                            
                if args.lloss:                    
                    sim_lloss = Loss_SimSiam(LossNet(base_model = args.base_model))

                    checkpoint = torch.load('sim_models/' + args.add_llosspretrained)

                    print('loading pretrained weights {}'.format(args.add_llosspretrained))

                    sim_lloss.load_state_dict(checkpoint['state_dict'])
            
            elif args.add_pretrained and self_method == 'encoder':

                auto_model = AutoEncoder(resnet.ResNet18(zero_init_residual=True))

                checkpoint = torch.load('sim_models/'+args.add_pretrained)

                print('loading pretrained weights {}'.format(args.add_pretrained))

                auto_model.load_state_dict(checkpoint['state_dict'])

            if args.initial:
                initial_sd = pickle.loads(pickle.dumps(copy.deepcopy(sim_model.state_dict())))
                initial_model = SimSiam(resnet.ResNet18(zero_init_residual=True))
                initial_model.load_state_dict(initial_sd)
                initial_model = initial_model.cuda()

                unlabeled_initial = indices[:SUBSET]
                fulldata_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetSequentialSampler(unlabeled_initial))

                initial_data = get_initial_kcg(initial_model.encoder, fulldata_loader, ADDENDUM, NUM_TRAIN)

        if args.total:
            labeled_set= indices
        
        elif args.initial:
            # labeled_set = indices[:ADDENDUM]
            # unlabeled_set = [x for x in indices if x not in labeled_set]
            labeled_set = initial_data[-ADDENDUM:]
            unlabeled_set = [x for x in initial_data if x not in labeled_set]
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True)
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        layers = ['layer4', 'layer3', 'layer2', 'layer1']
        
        for cycle in range(CYCLES):
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.base_model == 'resnet':
                    if args.dataset == "fashionmnist":
                        resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).to(args.device)
                    else:
                        # For self-supervised
                        if args.self_supervised and self_method == 'SimSiam':
                            new_model = SimSiam(resnet.ResNet18(zero_init_residual=True))

                            checkpoint = torch.load('sim_models/'+args.add_pretrained)

                            print('loading pretrained weights {}'.format(args.add_pretrained))

                            new_model.load_state_dict(checkpoint['state_dict'])

                            resnet18 = new_model.encoder.to(args.device)
                            # Reset model's fully connected layer
                            resnet18.fc = nn.Linear(512, NO_CLASSES).to(args.device)
                            
                            # Freeze Encoding part
                            if not args.frozen:
                                for name, param in resnet18.named_parameters():
                                    if name not in ['fc.weight', 'fc.bias']:
                                        param.requires_grad = False
                            
                            else:
                                if not args.layer == -1:
                                    for name, param in resnet18.named_parameters():
                                        print(name[:6])
                                        if name not in ['fc.weight', 'fc.bias'] and name[:6] not in layers[:args.layer]:
                                            param.requires_grad = False
                                else:
                                    pass

                            # initiate fully connected layer
                            resnet18.fc.weight.data.normal_(mean=0.0, std=0.01)
                            resnet18.fc.bias.data.zero_()

                            if args.lloss:
                                new_lloss = Loss_SimSiam(LossNet(base_model = args.base_model))

                                checkpoint = torch.load('sim_models/' + args.add_llosspretrained)

                                print('loading pretrained weights {}'.format(args.add_llosspretrained))

                                new_lloss.load_state_dict(checkpoint['state_dict'])

                                loss_module = new_lloss.encoder.to(args.device)

                                loss_module.linear = nn.Linear(4 * 128, 1).to(args.device)                                
                        
                        elif args.self_supervised and self_method == 'encoder':
                            new_model = AutoEncoder(resnet.ResNet18(zero_init_residual=True))

                            checkpoint = torch.load('sim_models/' + args.add_pretrained)
                            
                            print('loading pretrained weights {}'.format(args.add_pretrained))

                            new_model.load_state_dict(checkpoint['state_dict'])

                            resnet18 = new_model.encoder.to(args.device)
                            # Reset model's fully connected layer
                            resnet18.fc = nn.Linear(512, NO_CLASSES).to(args.device)
                            
                            # Freeze Encoding part
                            if not args.frozen:
                                for name, param in resnet18.named_parameters():
                                    if name not in ['fc.weight', 'fc.bias']:
                                        param.requires_grad = False
                            
                            # initiate fully connected layer
                            resnet18.fc.weight.data.normal_(mean=0.0, std=0.01)
                            resnet18.fc.bias.data.zero_()

                        else:
                            resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

                    if method == 'lloss' or method == 'TA-VAAL' and not args.lloss:
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
                else:
                    parameters = models['backbone'].parameters()
                    
                optim_backbone = optim.SGD(parameters, lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
    
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone}
                schedulers = {'backbone': sched_backbone}
                if method == 'lloss' or method == 'TA-VAAL':
                    optim_module   = optim.SGD(models['module'].parameters(), lr=LR_MODULE, 
                        momentum=MOMENTUM, weight_decay=WDECAY)
                    sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}
            else:
                ###Transformer
                models = {'backbone': transformer}
                if method =='lloss' or method == 'TA-VAAL':
                    models = {'backbone': transformer, 'module': loss_module}
                criterion      = nn.CrossEntropyLoss(reduction='none')
                optim_backbone = torch.optim.SGD(models['backbone'].parameters(),
                            lr=LR_TR,
                            momentum=0.9,
                            weight_decay=WDECAY)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                torch.backends.cudnn.benchmark = True

                optimizers = {'backbone': optim_backbone}
                schedulers = {'backbone': sched_backbone}                
                if method == 'lloss' or method == 'TA-VAAL':
                    optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                        momentum=MOMENTUM, weight_decay=WDECAY)
                    sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                    optimizers = {'backbone': optim_backbone, 'module': optim_module}
                    schedulers = {'backbone': sched_backbone, 'module': sched_module}                

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL, foldername, args, trial)
            acc = test(models, EPOCH, method, dataloaders, args, time, foldername,  len(labeled_set), trial, mode='test')
            if args.tsne:                
                unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetRandomSampler(subset), pin_memory=True)
                test_tsne(models, method, unlabeled_loader, dataloaders['train'], cycle, args, foldername, trial)

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
            # new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
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