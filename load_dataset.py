import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import *
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN, ImageNet
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def load_sim_dataset(dataset):

    img_size = 32
    if dataset == 'imagenet':
        img_size = 224

    augmentation = [
        T.RandomResizedCrop(img_size, scale=(0.2, 1.)),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]

    if dataset == 'cifar10': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=TwoCropsTransform(T.Compose(augmentation)))

    elif dataset == 'cifar10im': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=TwoCropsTransform(T.Compose(augmentation)))

    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=TwoCropsTransform(T.Compose(augmentation)))

    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=TwoCropsTransform(T.Compose(augmentation)))

    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=TwoCropsTransform(T.Compose(augmentation)))
    
    elif dataset == 'imagenet':
        data_train = ImageNet('../ImageNet', split='train', download=None, transform=TwoCropsTransform(T.Compose(augmentation)))

    return data_train



class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../cifar100', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../svhn', split="train", 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar10im":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag, 
                                    download=True, transform=transf)
            imbal_class_counts = [50, 5000] * 5
            targets = np.array(self.cifar10.targets)
            classes, class_counts = np.unique(targets, return_counts=True)
            nb_classes = len(classes)
            class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]
            # Get imbalanced number of instances
            imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
            imbal_class_indices = np.hstack(imbal_class_indices)

            # Set target and data to dataset
            self.cifar10.targets = targets[imbal_class_indices]
            self.cifar10.data = self.cifar10.data[imbal_class_indices]
        if self.dataset_name == 'imagenet':
            self.imagenet = ImageNet('../ImageNet', split='train',
                                        download=None, transform=transf)

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar10im":
            data, target = self.cifar10[index]            
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        if self.dataset_name == 'imagenet':
            data, target = self.imagenet[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar10im":
            return len(self.cifar10)        
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)
        elif self.dataset_name == 'imagenet':
            return len(self.imagenet)
##

# Data
def load_dataset(dataset):
    imagesize = 224 if dataset == 'imagenet' else 32
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=imagesize, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])


    if dataset == 'cifar10': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        NUM_TRAIN = len(data_train)        
        no_train = NUM_TRAIN
    elif dataset == 'cifar10im': 
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        #data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        NUM_TRAIN = targets.shape[0]
        classes, class_counts = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [50, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset(dataset, True, test_transform)
        #print(len(data_unlabeled))
        #data_unlabeled.cifar10.train_labels = targets[imb_class_idx]
        #data_unlabeled.cifar10.train_data = data_unlabeled.cifar10.train_data[imb_class_idx]
        data_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NUM_TRAIN = len(data_train)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test  = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
        adden = 2000
        NUM_TRAIN = len(data_train)        
        no_train = NUM_TRAIN
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = FashionMNIST('../fashionMNIST', train=False, download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)        
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test  = SVHN('../svhn', split='test', download=True, 
                                    transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        NUM_TRAIN = len(data_train)        
        adden = ADDENDUM
        no_train = NUM_TRAIN
    
    elif dataset == 'imagenet':
        data_train = ImageNet('../ImageNet', split='train', download=None,
                                            transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = ImageNet('../ImageNet', split='train', download=None,
                                            transform=test_transform)
        NO_CLASSES = 1000
        NUM_TRAIN = len(data_train)
        adden = 5000
        no_train = NUM_TRAIN
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train