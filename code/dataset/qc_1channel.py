# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
import skimage
from skimage import io, transform
import glob
import csv
import copy
import math
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
torch.backends.cudnn.benchmark = True



# PyTorch dataset class
class ImageQC_1channel_Dataset(torch.utils.data.Dataset):
    """
        ImageQC_1channel_Dataset class
    """
    # init method
    def __init__(self, files, img_size=2000, resize=True, is_train=True):
        """
        Args:
            files: path of the directory which contains the required images and labels
            img_size: resize the image to 'img_size' value
            resize: True or False
        """
        self.files = files
        self.img_size = img_size
        self.resize = resize
        self.is_train = is_train
        self.image_path = []
        self.labels = []
        
        # get the file paths and their corresponding labels
        self.image_path_0 = glob.glob(self.files + "/good/*")
        self.image_path_1 = glob.glob(self.files + "/blur/*")
        self.image_path_2 = glob.glob(self.files + "/empty/*")
        self.image_path_3 = glob.glob(self.files + "/debris/*")
        
        # append images and labels
        for index in range(0, len(self.image_path_0)):
            self.image_path.append(self.image_path_0[index])
            self.labels.append("0")
        
        for index in range(0, len(self.image_path_1)):
            self.image_path.append(self.image_path_1[index])
            self.labels.append("1")
            
        for index in range(0, len(self.image_path_2)):
            self.image_path.append(self.image_path_2[index])
            self.labels.append("2")
            
        for index in range(0, len(self.image_path_3)):
            self.image_path.append(self.image_path_3[index])
            self.labels.append("3")
                
                
    # getitem method
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        # read 1-channel image, if channels greater than 1 then take the first channel for the model(MPI and Pfizer partners had such issue)
        image = skimage.io.imread(self.image_path[index], plugin='pil')
        if len(image.shape) == 3 and int(image.shape[0]) >= 2 and int(image.shape[0]) <= 5:
            image = image[0, :, :]
        if len(image.shape) == 3 and int(image.shape[2]) >= 2 and int(image.shape[2]) <= 5:
            image = image[:, :, 0]
        
        # resize each channel and if image of size (5120x5120) then take the crop of (2000, 2000)(only Broad with Blur class had such issue)
        if self.resize:
            image = skimage.transform.resize(image, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
        if int(image.shape[0])>5000:
            image = image[2000:4000, 2000:4000]
        if self.is_train and int(image.shape[0])>2050:
            image = image[:2000, :2000]
        
        # normalization
        if float(image.max()) > 0.0:
            image = image / float(image.max())
            image = 255. * image
        image = torch.from_numpy(np.array(image).astype('uint8'))
        image = image / 255.
        image = image.unsqueeze(0)
        
        # return the image and the corresponding label
        if self.resize:
            return image, int(self.labels[index]), str(self.image_path[index])
        else:
            return image, int(self.labels[index]), int(image.shape[1]), str(self.image_path[index])
            
            
    # len method
    def __len__(self):
        """
        Returns:
            int: number of images in the directory
        """
        return len(self.labels)
    
    
# PyTorch class for Test dataset
class ImageQC_1channel_TestDataset(torch.utils.data.Dataset):
    """
        ImageQC_1channel_Dataset class
    """
    # init method
    def __init__(self, files, img_size=2000, resize=True):
        """
        Args:
            files: path of the directory which contains the required images and labels
            img_size: resize the image to 'img_size' value
            resize: True or False
        """
        self.files = files
        self.img_size = img_size
        self.resize = resize
        self.image_path = glob.glob(self.files + "/*")
                
                
    # getitem method
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        # read 1-channel image, if channels greater than 1 then take the first channel for the model(MPI and Pfizer partners had such issue)
        image = skimage.io.imread(self.image_path[index], plugin='pil')
        if len(image.shape) == 3 and int(image.shape[0]) >= 2 and int(image.shape[0]) <= 5:
            image = image[0, :, :]
        if len(image.shape) == 3 and int(image.shape[2]) >= 2 and int(image.shape[2]) <= 5:
            image = image[:, :, 0]
        
        # resize each channel and if image of size (5120x5120) then take the crop of (2000, 2000)(only Broad with Blur class had such issue)
        if self.resize:
            image = skimage.transform.resize(image, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
        if int(image.shape[0])>5000:
            image = image[2000:4000, 2000:4000]
        
        # normalization
        if float(image.max()) > 0.0:
            image = image / float(image.max())
            image = 255. * image
        image = torch.from_numpy(np.array(image).astype('uint8'))
        image = image / 255.
        image = image.unsqueeze(0)
        
        # return the image and the corresponding image path
        return image, str(self.image_path[index])
            
            
    # len method
    def __len__(self):
        """
        Returns:
            int: number of images in the directory
        """
        return len(self.image_path)
