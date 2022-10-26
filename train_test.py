from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import models.resnet as resnet
import torch.nn as nn
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import os
from collections import Counter
##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0)) # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    
    return loss


def test(models, epoch, method, dataloaders, args, time, foldername, datasize, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    deep_features = []
    actual = []
    num_cls = 10
    if args.dataset == 'cifar100':
        num_cls = 100

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, features, _ = models['backbone'](inputs)
            if args.tsne:
                deep_features += features.detach().cpu().numpy().tolist()
                actual += labels.detach().cpu().numpy().tolist()

            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        if args.tsne:
            tsne = TSNE(n_components=2)
            cluster = np.array(tsne.fit_transform(np.array(deep_features)))
            actual = np.array(actual)

            plt.figure(figsize=(20,20))
            for i in range(num_cls):
                idx = np.where(actual == i)
                plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=str(i))
            
            plt.legend()
            plt.savefig(foldername + '/' + 'results_' + time + '_' + str(args.method_type)+"_"+args.dataset +'_' + args.base_model + '_self-supervised' + str(args.self_supervised)+ 
            'datasize_' + str(datasize) + '_initial_'+ str(args.initial) + '_lr_' + str(args.lr) + '_frozen_' + str(args.frozen) + '_addednum_' + str(args.addednum) + '.png', dpi=200)

    return 100 * correct / total

def test_tsne(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'train'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    out_vec = torch.zeros(0)
    label = torch.zeros(0).long()
    with torch.no_grad():
        for (inputs, labels) in dataloaders:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            preds = scores.cpu()
            labels = labels.cpu()
            out_vec = torch.cat([out_vec,preds])
            label = torch.cat([label,labels])
        out_vec = out_vec.numpy()
        label = label.numpy()
    return out_vec,label

iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, foldername, num_epochs):

    models['backbone'].train()
    if method == 'lloss' or method == 'TA-VAAL':
        models['module'].train()
    
    class_distrib = os.path.join(foldername, 'classes.txt')
    
    class_file = open(class_distrib, 'a')    

    classes = []
    real_losses = []
    pred_losses = []

    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        
        target_loss = criterion(scores, labels)
        if method == 'lloss' or method == 'TA-VAAL':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].step()
        
        if (epoch+1) == num_epochs:
            if method == 'lloss' or method == 'TA-VAAL':
                real_losses += target_loss.detach().cpu().numpy().tolist()
                pred_losses += pred_loss.detach().cpu().numpy().tolist()
            classes +=  labels.detach().cpu().numpy().tolist()
    
    if (epoch+1) == num_epochs:
        counter = Counter()
        counter.update(classes)
        class_file.write("{} : {}\n".format(len(classes), dict(counter)))
        if method == 'lloss' or method == 'TA-VAAL':
            loss_file = os.path.join(foldername, '{}_loss.csv'.format(len(classes)))
            df = pd.DataFrame()
            df['real_losses'] = real_losses
            df['pred_losses'] = pred_losses

            df.to_csv(loss_file, mode='a')    

    return loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, foldername):
    print('>> Train a Model.')
    best_acc = 0.
        
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss, foldername, num_epochs)

        schedulers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            schedulers['module'].step()

        if False and epoch % 20  == 7:
            acc = test(models, epoch, method, dataloaders, mode='test')
            # acc = test(models, dataloaders, mc, 'test')
            if best_acc < acc:
                best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
