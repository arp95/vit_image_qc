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
class ImageQC_5channel_Dataset(torch.utils.data.Dataset):
    """
        ImageQC_5channel_Dataset.
    """
    # init method
    def __init__(self, files, img_size=1024, resize=True):
        """
        Args:
            files: path of the directory which contains the required images and labels
            img_size: resize the image to 'img_size' value
            resize: True or False
        """
        self.files = files
        self.img_size = img_size
        self.resize = resize
        
        # get the file paths and their corresponding labels(0: good, 1: blurry, 2: empty, 3: debris)
        self.labels_0 = []
        self.image_path_dna_0 = []
        self.image_path_rna_0 = []
        self.image_path_er_0 = []
        self.image_path_mito_0 = []
        self.image_path_agp_0 = []
        
        self.labels_1 = []
        self.image_path_dna_1 = []
        self.image_path_rna_1 = []
        self.image_path_er_1 = []
        self.image_path_mito_1 = []
        self.image_path_agp_1 = []
        
        self.labels_2 = []
        self.image_path_dna_2 = []
        self.image_path_rna_2 = []
        self.image_path_er_2 = []
        self.image_path_mito_2 = []
        self.image_path_agp_2 = []
        
        self.labels_3 = []
        self.image_path_dna_3 = []
        self.image_path_rna_3 = []
        self.image_path_er_3 = []
        self.image_path_mito_3 = []
        self.image_path_agp_3 = []
        
        self.labels = []
        self.image_path_dna = []
        self.image_path_rna = []
        self.image_path_er = []
        self.image_path_mito = []
        self.image_path_agp = []
        for file in files:
            flag = -1
    
            with open(file, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    if flag == -1:
                        flag = 1
                    else:
                        array = row
                        fixed_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots"
                        dna_path = array[0].split("/")
                        rna_path = array[1].split("/")
                        er_path = array[2].split("/")
                        mito_path = array[3].split("/")
                        agp_path = array[4].split("/")
                        
                        if dna_path[1] == "dgx1nas1":
                            if int(array[5]) == 0:
                                self.image_path_dna_0.append(str(array[0]))
                                self.image_path_rna_0.append(str(array[1]))
                                self.image_path_er_0.append(str(array[2]))
                                self.image_path_mito_0.append(str(array[3]))
                                self.image_path_agp_0.append(str(array[4]))
                                self.labels_0.append(str(array[5]))
                            elif int(array[5]) == 1:
                                self.image_path_dna_1.append(str(array[0]))
                                self.image_path_rna_1.append(str(array[1]))
                                self.image_path_er_1.append(str(array[2]))
                                self.image_path_mito_1.append(str(array[3]))
                                self.image_path_agp_1.append(str(array[4]))
                                self.labels_1.append(str(array[5]))
                            elif int(array[5]) == 2:
                                self.image_path_dna_2.append(str(array[0]))
                                self.image_path_rna_2.append(str(array[1]))
                                self.image_path_er_2.append(str(array[2]))
                                self.image_path_mito_2.append(str(array[3]))
                                self.image_path_agp_2.append(str(array[4]))
                                self.labels_2.append(str(array[5]))
                            elif int(array[5]) == 3:
                                self.image_path_dna_3.append(str(array[0]))
                                self.image_path_rna_3.append(str(array[1]))
                                self.image_path_er_3.append(str(array[2]))
                                self.image_path_mito_3.append(str(array[3]))
                                self.image_path_agp_3.append(str(array[4]))
                                self.labels_3.append(str(array[5]))
                        else:
                            new_dna_path = fixed_path
                            new_rna_path = fixed_path
                            new_er_path = fixed_path
                            new_mito_path = fixed_path
                            new_agp_path = fixed_path
                            for index in range(3, len(dna_path)):
                                new_dna_path = new_dna_path + "/" + dna_path[index]
                                new_rna_path = new_rna_path + "/" + rna_path[index]
                                new_er_path = new_er_path + "/" + er_path[index]
                                new_mito_path = new_mito_path + "/" + mito_path[index]
                                new_agp_path = new_agp_path + "/" + agp_path[index]
                                
                            if int(array[5]) == 0:
                                self.image_path_dna_0.append(str(new_dna_path))
                                self.image_path_rna_0.append(str(new_rna_path))
                                self.image_path_er_0.append(str(new_er_path))
                                self.image_path_mito_0.append(str(new_mito_path))
                                self.image_path_agp_0.append(str(new_agp_path))
                                self.labels_0.append(str(array[5]))
                            elif int(array[5]) == 1:
                                self.image_path_dna_1.append(str(new_dna_path))
                                self.image_path_rna_1.append(str(new_rna_path))
                                self.image_path_er_1.append(str(new_er_path))
                                self.image_path_mito_1.append(str(new_mito_path))
                                self.image_path_agp_1.append(str(new_agp_path))
                                self.labels_1.append(str(array[5]))
                            elif int(array[5]) == 2:
                                self.image_path_dna_2.append(str(new_dna_path))
                                self.image_path_rna_2.append(str(new_rna_path))
                                self.image_path_er_2.append(str(new_er_path))
                                self.image_path_mito_2.append(str(new_mito_path))
                                self.image_path_agp_2.append(str(new_agp_path))
                                self.labels_2.append(str(array[5]))
                            elif int(array[5]) == 3:
                                self.image_path_dna_3.append(str(new_dna_path))
                                self.image_path_rna_3.append(str(new_rna_path))
                                self.image_path_er_3.append(str(new_er_path))
                                self.image_path_mito_3.append(str(new_mito_path))
                                self.image_path_agp_3.append(str(new_agp_path))
                                self.labels_3.append(str(array[5]))
                                
        # get good, empty and debris images
        for index1 in range(0, len(self.labels_0)):
            self.image_path_dna.append(self.image_path_dna_0[index1])
            self.image_path_rna.append(self.image_path_rna_0[index1])
            self.image_path_er.append(self.image_path_er_0[index1])
            self.image_path_mito.append(self.image_path_mito_0[index1])
            self.image_path_agp.append(self.image_path_agp_0[index1])
            self.labels.append("0")
        
        for index1 in range(0, len(self.labels_1)):
            self.image_path_dna.append(self.image_path_dna_1[index1])
            self.image_path_rna.append(self.image_path_rna_1[index1])
            self.image_path_er.append(self.image_path_er_1[index1])
            self.image_path_mito.append(self.image_path_mito_1[index1])
            self.image_path_agp.append(self.image_path_agp_1[index1])
            self.labels.append("1")
        
        for index1 in range(0, len(self.labels_2)):
            self.image_path_dna.append(self.image_path_dna_2[index1])
            self.image_path_rna.append(self.image_path_rna_2[index1])
            self.image_path_er.append(self.image_path_er_2[index1])
            self.image_path_mito.append(self.image_path_mito_2[index1])
            self.image_path_agp.append(self.image_path_agp_2[index1])
            self.labels.append("2")
            
        for index1 in range(0, len(self.labels_3)):
            self.image_path_dna.append(self.image_path_dna_3[index1])
            self.image_path_rna.append(self.image_path_rna_3[index1])
            self.image_path_er.append(self.image_path_er_3[index1])
            self.image_path_mito.append(self.image_path_mito_3[index1])
            self.image_path_agp.append(self.image_path_agp_3[index1])
            self.labels.append("3")
                                
                
    # getitem method
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        # read each channel
        self.image_path_dna[index] = self.image_path_dna[index].replace("%20", " ")
        self.image_path_rna[index] = self.image_path_rna[index].replace("%20", " ")
        self.image_path_er[index] = self.image_path_er[index].replace("%20", " ")
        self.image_path_mito[index] = self.image_path_mito[index].replace("%20", " ")
        self.image_path_agp[index] = self.image_path_agp[index].replace("%20", " ")
        image_dna = skimage.io.imread(self.image_path_dna[index], plugin='pil')
        image_rna = skimage.io.imread(self.image_path_rna[index], plugin='pil')
        image_er = skimage.io.imread(self.image_path_er[index], plugin='pil')
        image_mito = skimage.io.imread(self.image_path_mito[index], plugin='pil')
        image_agp = skimage.io.imread(self.image_path_agp[index], plugin='pil')
        
        # resize each channel
        if self.resize:
            image_dna = skimage.transform.resize(image_dna, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
            image_rna = skimage.transform.resize(image_rna, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
            image_er = skimage.transform.resize(image_er, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
            image_mito = skimage.transform.resize(image_mito, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
            image_agp = skimage.transform.resize(image_agp, [self.img_size, self.img_size], mode='constant', preserve_range=True, order=0)
        
        # convert each channel to type uint8
        image_dna = image_dna / float(image_dna.max())
        image_dna = 255. * image_dna
        image_rna = image_rna / float(image_rna.max())
        image_rna = 255. * image_rna
        image_er = image_er / float(image_er.max())
        image_er = 255. * image_er
        image_mito = image_mito / float(image_mito.max())
        image_mito = 255. * image_mito
        image_agp = image_agp / float(image_agp.max())
        image_agp = 255. * image_agp
        
        # convert them to PyTorch tensor
        image_dna = torch.from_numpy(np.array(image_dna).astype('uint8'))
        image_rna = torch.from_numpy(np.array(image_rna).astype('uint8'))
        image_er = torch.from_numpy(np.array(image_er).astype('uint8'))
        image_mito = torch.from_numpy(np.array(image_mito).astype('uint8'))
        image_agp = torch.from_numpy(np.array(image_agp).astype('uint8'))
        
        # normalize the images and merge
        image_dna = image_dna / 255.
        image_rna = image_rna / 255.
        image_er = image_er / 255.
        image_mito = image_mito / 255.
        image_agp = image_agp / 255.
        image = torch.stack((image_dna, image_er, image_rna, image_agp, image_mito))
        
        # return the image and the corresponding label
        if self.resize:
            return image, int(self.labels[index])
        else:
            img = torch.zeros(1, 2160, 2160)
            img[:, :int(image.shape[1]), :int(image.shape[2])] = image
            return img, int(self.labels[index]), int(image.shape[1])

        
    # len method
    def __len__(self):
        """
        Returns:
            int: number of images in the directory
        """
        return len(self.labels)
