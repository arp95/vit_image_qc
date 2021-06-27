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
from dataset import *


# dataset paths
is_pretrained = True
image_size = 1024
val_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/2021_03_03_Stain5_CondC_PE_Standard/images/BR00120277__2021-02-20T07_02_46-Measurement1/Images"
model_path = "/home/jupyter-arpit@broadinstitu-ef612/qc_bestmodel_baseline.pth"
output_path = "/home/jupyter-arpit@broadinstitu-ef612/test_baseline_5.csv"


# create PyTorch dataset class and create val_data and the val_loader
val_data = ImageQC_1channel_TestDataset(val_path, img_size=image_size)
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
    torch.nn.Linear(512, 4),
    torch.nn.Softmax()
)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


# test loop
model.eval()
file_path = []
file_labels = []
file_good_prob = []
file_blurry_prob = []
file_empty_prob = []
file_debris_prob = []
with torch.no_grad():
    for i, (input, image_path) in enumerate(val_loader):
        input = input.to(device)
        output = model(input)
        output_probs = output[0]
        _, predicted = output.max(1)
        
        # append the results
        file_path.append(image_path[0])
        file_labels.append(int(predicted[0]))
        file_good_prob.append(round(float(output_probs[0]), 3))
        file_blurry_prob.append(round(float(output_probs[1]), 3))
        file_empty_prob.append(round(float(output_probs[2]), 3))
        file_debris_prob.append(round(float(output_probs[3]), 3))


# write results
with open(output_path, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["File Path", "Good Prob", "Blurry Prob", "Empty Prob", "Debris Prob", "Label"])
    for index in range(0, len(file_path)):
        spamwriter.writerow([str(file_path[index]), file_good_prob[index], file_blurry_prob[index], file_empty_prob[index], file_debris_prob[index], file_labels[index]])
        
print()
print("Done with processing!")
