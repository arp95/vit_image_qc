# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
import skimage
from skimage import io, transform
import glob
from PIL import Image
from PIL import ImageFile
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataset import *
from metrics import *


# dataset paths
is_pretrained = True
image_size = 1024
model_path = "/home/jupyter-arpit@broadinstitu-ef612/qc_bestmodel_baseline_size2_pretrained.pth"
val_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/validation"


# create PyTorch dataset class and create val_data and the val_loader
val_data = ImageQC_1channel_Dataset(val_path, img_size=image_size, is_train=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
print(len(val_data))


# define model and load it to gpu or cpu
model = torchvision.models.resnet50(pretrained=is_pretrained)
if is_pretrained:
    for param in model.parameters():
        param.requires_grad = False
model.conv1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    torch.nn.BatchNorm2d(3),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(512, 4)
)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


# validation loop
metrics = StreamMetrics(4)
metrics.reset()
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for i, (input, target, image_path) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
                
        # get confusion matrix
        targets = target.cpu().numpy()
        predicted = predicted.cpu().numpy()
        metrics.update(targets, predicted)
            

# print results
valid_accuracy = str(100.0*(float(correct)/float(total)))
results = metrics.get_results()
confusion_matrix = results["Confusion Matrix"]
print()
print("Validation Accuracy: " + str(valid_accuracy))
print(confusion_matrix)
print()
