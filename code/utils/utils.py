# header files
import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np
import skimage
from skimage import io, transform
import glob
import csv
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
torch.backends.cudnn.benchmark = True


def collate_fn_1channel(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    input, labels, lengths, _ = zip(*data)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    input = list(input)
    
    # get resolution of debris class or blurry class
    max_debris_res = -1
    for i in range(4):
        if labels[i] == 3:
            max_debris_res = int(lengths[i])
        '''
        elif labels[i] == 3:
            find_label = -1
            find_sample = -1
            find_length = -1
            for j in range(4):
                if labels[j] != 3:
                    find_label = labels[j]
                    find_sample = input[j]
                    find_length = lengths[j]
            if find_label != -1:
                labels[i] = find_label
                lengths[i] = find_length
                input[i] = find_sample
        '''
    if max_debris_res == -1:
        max_debris_res = int(max(lengths))
        
    # create batch of images with debris max resolution
    features = torch.zeros(4, 1, int(max_debris_res), int(max_debris_res))
    for i in range(4):
        sample = input[i]
        if max_debris_res < int(lengths[i]):
            features[i] = sample[:, :int(max_debris_res), :int(max_debris_res)]
        else:
            features[i, :, :int(lengths[i]), :int(lengths[i])] = sample
    return features, labels.long(), lengths.long()


# class: handles class imbalance during training 
class ImbalancedDatasetSampler_1channel(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        dataset: the list of train_data
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
        weights: the list of weights for each sample
        label_weights: the number of samples for each class
    """

    
    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None, weights=None, label_weights=None):
                
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        label_to_count[0] = label_weights[0]
        label_to_count[1] = label_weights[1]
        label_to_count[2] = label_weights[2]
        label_to_count[3] = label_weights[3]
                
        # weight for each sample
        if weights == None:
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            return dataset.__getitem__(idx)[1]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
# handling class imbalance during training 
class ImbalancedDatasetSampler_5channel(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None, weights=None, label_weights=None):
                
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        label_to_count[0] = label_weights[0]
        label_to_count[1] = label_weights[1]
        label_to_count[2] = label_weights[2]
        label_to_count[3] = label_weights[2]
                
        # weight for each sample
        if weights == None:
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            return dataset.__getitem__(idx)[1]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
# class: Cross-Entropy loss with Label Smoothing
class CrossEntropyLabelSmoothingLoss(torch.nn.Module):
    """Cross-Entropy loss with Label Smoothing
    Arguments:
        smoothing: the smoothing factor lies between 0 and 1
    """
    
    
    def __init__(self, smoothing=0.0):
        super(CrossEntropyLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        weight = input.new_ones(pred.size()) * (self.smoothing/(pred.size(-1)-1.))
        weight.scatter_(-1, target.unsqueeze(-1), (1.-self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
    
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
