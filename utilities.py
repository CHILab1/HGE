import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm, GCNConv, GraphConv, SGConv, LayerNorm
from torch_geometric.nn import global_mean_pool, global_add_pool
import pickle
from util import classification_process
import os
from datetime import datetime
import random 
import sys
sys.path.append("/home/scontino/python/graph_vae/")
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import ast
import numpy as np
from Package.Train_metrics.Metrics import *
from Package.DataProcessing.DataProcessing import *
import time
from torch.autograd import Variable
import pandas as pd
import ast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from model import *



class Weighted_BCE_loss(nn.Module):
    """
        Binary cross-entropy loss function with weighted labels.
        :param y_pred: predicted values (as a tensor)
        :param y_true: true values (as a tensor)
        :param pos_weight: weight for positive labels (as a float)
        :param neg_weight: weight for negative labels (as a float)
        :return: binary cross-entropy loss (as a tensor)
    """
    def __init__(self, pos_weight, neg_weight):
        super(Weighted_BCE_loss, self).__init__()e
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(self, y_pred, y_true):
        # Compute the binary cross-entropy loss for each element
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        # Apply the weights to the loss
        weighted_bce_loss = y_true * self.pos_weight * bce_loss + (1 - y_true) * self.neg_weight * bce_loss
        # Compute the mean of the weighted loss over all elements
        mean_loss = torch.mean(weighted_bce_loss)
        return mean_loss
