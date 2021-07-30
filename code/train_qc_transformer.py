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
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
from dataset import *
from utils import *
from metrics import *
from model import *
torch.backends.cudnn.benchmark = True

# ensure same result is produced
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


# dataset paths
train_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/train"
val_path = "/dgx1nas1/cellpainting-datasets/2019_07_11_JUMP_CP_pilots/validation"
results_dir = "/home/jupyter-arpit@broadinstitu-ef612/"
gpu_on_dgx = "cuda:4"


# hyperparameters
lr = 0.005
batch_size = 4
num_epochs = 151
output_classes = 4


# create PyTorch dataset class and create train data and val data
train_data = ImageQC_1channel_Dataset(train_path, resize=False, is_train=True)
val_data = ImageQC_1channel_Dataset(val_path, resize=False, is_train=False)
print(len(train_data))
print(len(val_data))


# load the data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_1channel)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=10)


# model
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

# load model to gpu
device = torch.device(gpu_on_dgx if torch.cuda.is_available() else "cpu")
model = VisionTransformer(config=get_config(), num_classes=output_classes, in_channels=1)
model.to(device)


# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=int(10*int(len(train_loader))), max_iters=int(num_epochs*int(len(train_loader))))


# define loss (smoothing=0 is equivalent to standard Cross-Entropy loss)
criterion = torch.nn.CrossEntropyLoss()


# train and validate loop
metrics = StreamMetrics(output_classes)
best_metric = -1
best_metric_epoch = -1
train_loss = []
val_loss = []
train_acc = []
val_acc = []
confusion_matrix = None
best_confusion_matrix = -1

# train and validate
for epoch in range(1, num_epochs+50):
    print("Epoch: " + str(epoch))
    print()
    
    # train
    model.train()
    training_loss = 0.0
    total = 0
    correct = 0
    for i, (input, target, lengths) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(input)[0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        training_loss = training_loss + loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    training_loss = training_loss/float(len(train_loader))
    training_accuracy = str(100.0*(float(correct)/float(total)))
    train_acc.append(training_accuracy)
    train_loss.append(training_loss)
    
    # validate
    if epoch%5 == 0:
        metrics.reset()
        model.eval()
        valid_loss = 0.0
        total = 0
        correct = 0
        for i, (input, target, lengths, _) in enumerate(val_loader):
            with torch.no_grad():
                input = input.to(device)
                target = target.to(device)
                lengths = lengths.to(device)

                output = model(input)[0]
                loss = criterion(output, target)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # get confusion matrix
                targets = target.cpu().numpy()
                predicted = predicted.cpu().numpy()
                metrics.update(targets, predicted)
            valid_loss = valid_loss + loss.item()
        valid_loss = valid_loss/float(len(val_loader))
        valid_accuracy = str(100.0*(float(correct)/float(total)))
        results = metrics.get_results()
        confusion_matrix = results["Confusion Matrix"]
        val_loss.append(valid_loss)
        val_acc.append(valid_accuracy)

        # store best model
        if(float(valid_accuracy)>best_metric):
            best_metric = float(valid_accuracy)
            best_metric_epoch = epoch
            best_confusion_matrix = confusion_matrix
            torch.save(model.state_dict(), results_dir + "qc_bestmodel_transformer.pth")
    
        print()
        print("Epoch" + str(epoch) + ":")
        print("Training Accuracy: " + str(training_accuracy) + "    Validation Accuracy: " + str(valid_accuracy))
        print("Training Loss: " + str(training_loss) + "    Validation Loss: " + str(valid_loss))
        print("Best metric: " + str(best_metric))
        print(confusion_matrix)
        print(best_confusion_matrix)
        print()
    
    # lr scheduler
    if epoch <= 151:
        lr_scheduler.step()
    
    
# val_loss vs epoch
epoch = []
for i in range(0, len(val_acc)):
    epoch.append(i*5)
    val_acc[i] = float(val_acc[i])
    val_loss[i] = float(val_loss[i])
    
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.plot(epoch, val_loss)
plt.savefig(results_dir + "val_loss_qc_transformer.png")

# val_acc vs epoch
plt.cla()
epoch = []
for i in range(0, len(val_acc)):
    epoch.append(i*5)
    val_acc[i] = float(val_acc[i])
    val_loss[i] = float(val_loss[i])
    
plt.xlabel("Epochs")
plt.ylabel("Validation Acc")
plt.plot(epoch, val_acc)
plt.savefig(results_dir +  "val_acc_qc_transformer.png")
