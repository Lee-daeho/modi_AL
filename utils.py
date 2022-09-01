import torch
import shutil
import numpy as np
from kcenterGreedy import kCenterGreedy
from config import *
from kmeans_pytorch import kmeans, pairwise_distance

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_initial_kcg(model, unlabeled_loader, addednums, train_num):
    model.eval()
    mins = np.array([float('inf')] * train_num)
    batch = [0] * train_num
    with torch.cuda.device(0):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            with torch.cuda.device(0):
                inputs = inputs.cuda()
            _, features_batch, _ = model(inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.cpu().detach()
        new_av_idx = np.array([])
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, addednums)
        other_idx = [x for x in range(train_num) if x not in batch]

    return  other_idx + batch
