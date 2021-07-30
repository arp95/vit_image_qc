# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
import skimage
from skimage import io, transform
import glob
from PIL import Image
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
from dataset import *
from utils import *
from metrics import *
from model import *


# dataset and model paths
model_path = "/home/jupyter-arpit@broadinstitu-ef612/qc_bestmodel_transformer.pth"
val_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/validation"
gpu_on_dgx = "cuda:4"


# create PyTorch dataset class and create val data and val_loader
val_data = ImageQC_1channel_Dataset(val_path, resize=False, is_train=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
print(len(val_data))


# define model and load it to cpu/gpu
def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 32
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 64
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

device = torch.device(gpu_on_dgx if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config=get_config(), num_classes=4, in_channels=1)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


# validation loop
metrics = StreamMetrics(4)
metrics.reset()
model.eval()
total = 0
correct = 0
for i, (input, target, lengths, image_path) in enumerate(val_loader):
    with torch.no_grad():
        input = input.to(device)
        target = target.to(device)
        lengths = lengths.to(device)
        
        output = model(input)[0]
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
                
        # get confusion matrix
        targets = target.cpu().numpy()
        predicted = predicted.cpu().numpy()
        metrics.update(targets, predicted)
        
        #if targets[0] == 0 and predicted[0] != 0:
        #    print(predicted[0])
        #    print(image_path)
        
valid_accuracy = str(100.0*(float(correct)/float(total)))
results = metrics.get_results()
confusion_matrix = results["Confusion Matrix"]
print()
print("Validation Accuracy: " + str(valid_accuracy))
print(confusion_matrix)
print()
