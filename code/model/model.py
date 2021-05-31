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
import ml_collections
torch.backends.cudnn.benchmark = True


# class: Resnet50
class Resnet50(torch.nn.Module):
    def __init__(self, output_layer='layer4', in_channels=5, is_pretrained=True):
        super().__init__()
        
        self.output_layer = output_layer
        self.pretrained = torchvision.models.resnet50(pretrained=is_pretrained)
        if is_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False
           
        # change from 5-channel image to 3-channel image
        self.pretrained.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        )
        
        # get last layer and apply feature projection layer that converts number of features maps to 32
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break
        self.resnet = nn.Sequential(*self.children_list)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        )
        
        # adaptive avg-pool to get final feature map size of (32x32x32)
        self.adaptive_avgpool = torch.nn.AdaptiveAvgPool2d((32, 32))
        
    def forward(self, x, lengths=None):
        x = self.resnet(x)
        x = self.feature_projection(x)
        x = self.adaptive_avgpool(x)
        return x
    
    
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)


# class: Attention
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# class: Mlp
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# class: Embeddings
class Embeddings(nn.Module):
    """
        Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=5, is_pretrained=True, is_hybrid=True):
        super(Embeddings, self).__init__()
        
        self.is_hybrid = is_hybrid
        self.hybrid_model = Resnet50(in_channels=in_channels, is_pretrained=is_pretrained)
        self.patch_embeddings = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, config.hidden_size, kernel_size=16, stride=16)
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1025, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, lengths=None):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if self.is_hybrid:
            x = self.hybrid_model(x, lengths=lengths)
        else:
            x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# class: Block
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


# class: Encoder
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# class: Transformer
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, in_channels=5, is_pretrained=True, is_hybrid=True):
        super(Transformer, self).__init__()
        
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=in_channels, is_pretrained=is_pretrained, is_hybrid=is_hybrid)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, lengths=None):
        embedding_output = self.embeddings(input_ids, lengths=lengths)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


# main class: VisionTransformer
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=32, num_classes=4, zero_head=False, vis=False, in_channels=5, is_pretrained=True, is_hybrid=True):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, in_channels=in_channels, is_pretrained=is_pretrained, is_hybrid=is_hybrid)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, lengths=None):
        x, attn_weights = self.transformer(x, lengths)
        logits = self.head(x[:, 0])
        return logits, attn_weights
