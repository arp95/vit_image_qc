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
from PIL import ImageFile
from dataset import *
from utils import *
from model import *


# dataset and model paths
val_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/2021_03_03_Stain5_CondC_PE_Standard/images/BR00120277__2021-02-20T07_02_46-Measurement1/Images"
model_path = "/home/jupyter-arpit@broadinstitu-ef612/qc_bestmodel_transformer.pth"
output_path = "/home/jupyter-arpit@broadinstitu-ef612/test_transformer_batch.csv"
gpu_on_dgx = "cuda:4"


# create PyTorch dataset class and create val data and val_loader
val_data = ImageQC_1channel_TestDataset(val_path, resize=False)
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
model.eval()
file_path = []
file_labels = []
file_good_prob = []
file_blurry_prob = []
file_empty_prob = []
file_debris_prob = []
for i, (input, image_path) in enumerate(val_loader):
    with torch.no_grad():
        input = input.to(device)
        output = model(input)[0]
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
