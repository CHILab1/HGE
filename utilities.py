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

class EarlyStopping:
    def __init__(self, dir, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_dir = dir
    
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            torch.save(model.state_dict(), self.save_dir + "model_best_loss.pth")
        elif val_loss > self.best_score:
            self.counter += 1
            print("Early stopping counter: ", self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            torch.save(model.state_dict(), self.save_dir + "model_best_loss.pth")
            self.counter = 0

class MetricWatcher:
    def __init__(self, dir):
        self.best_pos = 0
        self.best_neg = 0
        self.save_dir = dir
    
    def __call__(self, neg, pos, model):
        if self.best_pos == 0 or self.best_neg == 0:
            self.best_pos = pos
            self.best_neg = neg
        if neg > self.best_neg and pos > self.best_pos:
            self.best_pos = pos
            self.best_neg = neg
            print("Saving best metric model...")
            torch.save(model.state_dict(), self.save_dir + "model_best_metric.pth")

class F2Loss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self):
        super(F2Loss, self).__init__()

    def forward(self, preds, labels, device, grad=True):
        total_loss = torch.tensor(0, dtype=torch.float).to(device)
        f2 = torch.tensor(0, dtype=torch.float).to(device) 
        for p, l in zip(preds, labels):
            TP, FP, TN, FN = confusion(l, torch.round(p), device)

            precision = TP / (TP + FP + 1e-12)
            recall = TP / (TP + FN + 1e-12)
            f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-12)
            total_loss += (1 - f2)
        return total_loss

class softFscore:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __call__(self, y_pred, y_true):
        tp = torch.sum(y_pred * y_true, dim=0)
        fp = torch.sum(y_pred * (1 - y_true), dim=0)
        fn = torch.sum((1 - y_pred) * y_true, dim=0)
        soft_fscore = ((1 + (self.beta**2)) * tp) / (((1 + (self.beta**2)) *tp) + (self.beta**2)*fn + fp + 1e-16)
        soft_fscore = 1 - soft_fscore
        return torch.mean(soft_fscore)
        # return soft_fscore

# f2 score from here: https://discuss.pytorch.org/t/create-a-f-score-loss-function/102279
class F2BCELoss(nn.Module):
    "Loss F2 Score + BCE"
    def __init__(self, eps=1e-7, balance_factor=0.5, device=None) -> None:
        super().__init__()
        self.eps = eps
        self.balance_factor = balance_factor
        self.focal = WeightedFocalLoss(alpha=0.25, gamma=2, device=device)
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, beta=2, fp_factor=1.5, grad=True):
        focal = self.focal(y_pred, y_true)
        mse = self.mse(y_pred, y_true)
        tp = (y_true * y_pred).sum().to(torch.float32)
        #! versione vecchia secondo me sbagliata
        #! versione modificata secondo me giusta
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f_score_loss = (1 + beta ** 2) * (precision * recall) / ((beta**2)*precision + recall + self.eps)
        total_loss = (self.balance_factor) * (1 - f_score_loss) + (1 - self.balance_factor) * (focal)
        return total_loss

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

class F2wBCELoss(nn.Module):
    "Loss F2 Score + BCE"
    def __init__(self, wbce_pos_weights, wbce_neg_weights, f_balance_factor=0.5, beta=2., eps=1e-7):
        super(F2wBCELoss, self).__init__()
        self.eps = eps
        self.beta = beta
        self.balance_factor = f_balance_factor
        self.wbce = Weighted_BCE_loss(pos_weight=wbce_pos_weights, neg_weight=wbce_neg_weights)

    def forward(self, y_pred, y_true, fp_factor=1.5, grad=True):
        wbce = self.wbce(y_pred, y_true)

        y_pred = torch.round(y_pred)
        tp = (y_true * y_pred).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f_score_loss = (1 + self.beta ** 2) * (precision * recall) / ((self.beta**2)*precision + recall + self.eps)
        total_loss = (self.balance_factor) * (1 - f_score_loss) + (1 - self.balance_factor) * (wbce)
        return total_loss

class DoubleFbetawBCELoss(nn.Module):
    "Loss F2 Score + BCE"
    def __init__(self, wbce_pos_weights, wbce_neg_weights, f_balance_factor=0.5, beta=2, eps=1e-7):
        super(DoubleFbetawBCELoss, self).__init__()
        self.eps = eps
        self.beta = beta
        self.balance_factor = f_balance_factor
        self.wbce = Weighted_BCE_loss(pos_weight=wbce_pos_weights, neg_weight=wbce_neg_weights)

    def forward(self, y_pred, y_true, fp_factor=1.5, grad=True):
        wbce = self.wbce(y_pred, y_true)

        y_pred = torch.round(y_pred)
        tp = (y_true * y_pred).sum().to(torch.float32)

        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f_score_loss = (1 + self.beta ** 2) * (precision * recall) / ((self.beta**2)*precision + recall + self.eps)
        aux_f_score_loss = (1 + 0.5 ** 2) * (precision * recall) / ((0.5**2)*precision + recall + self.eps)

        total_loss = (self.balance_factor) * ((1 - f_score_loss) + (1 - aux_f_score_loss)) + (1 - self.balance_factor) * (wbce)
        return total_loss

class SoftFbetawBCELoss(nn.Module):
    "Loss F2 Score + BCE"
    def __init__(self, wbce_pos_weights, wbce_neg_weights, f_balance_factor=0.5, beta=2., eps=1e-7):
        super(SoftFbetawBCELoss, self).__init__()
        self.eps = eps
        self.balance_factor = f_balance_factor
        self.softFscore = softFscore(beta=beta)
        self.wbce = Weighted_BCE_loss(pos_weight=wbce_pos_weights, neg_weight=wbce_neg_weights)

    def forward(self, y_pred, y_true):
        wbce = self.wbce(y_pred, y_true)
        softFscore = self.softFscore(y_pred, y_true)
        total_loss = (self.balance_factor) * (softFscore) + (1 - self.balance_factor) * (wbce)
        return total_loss

class SoftFbetaBCELoss(nn.Module):
    "Loss F2 Score + BCE"
    def __init__(self, f_balance_factor=0.5, beta=2., eps=1e-7):
        super(SoftFbetaBCELoss, self).__init__()
        self.eps = eps
        self.balance_factor = f_balance_factor
        self.softFscore = softFscore(beta=beta)
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        bce = self.bce(y_pred, y_true)
        softFscore = self.softFscore(y_pred, y_true)
        total_loss = (self.balance_factor) * (softFscore) + (1 - self.balance_factor) * (bce)
        return total_loss

def f2_loss(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    total_loss = torch.tensor(0, dtype=torch.float)
    for p, l in zip(preds, labels):
        confusion = confusion_matrix(l, np.round(p))
        if confusion.shape == (1, 1):
            print(confusion)
            print("label", l)
            print("prediction", p)
        TN = torch.tensor(confusion[0, 0])
        FN = torch.tensor(confusion[0, 1])
        FP = torch.tensor(confusion[1, 0])
        TP = torch.tensor(confusion[1, 1])
        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        f2 = 5 * (precision * recall) / (4 * precision + recall + 1e-12)
        total_loss += (1 - f2.float())
    return total_loss

def balanced_dataset(graphs, crop_index):
    positive_g = []
    positive_counter = 0
    negative_g = []
    negative_counter = 0
    for g in graphs:
        if g.y == 0 and (negative_counter < (crop_index/2)):
            negative_g.append(g)
            negative_counter+=1
        if g.y == 1 and (positive_counter < (crop_index/2)):
            positive_g.append(g)
            positive_counter+=1
        if negative_counter == (crop_index/2) and positive_counter == (crop_index/2):
            break
    print("len positive graphs: ", len(positive_g))
    print("len negative graphs: ", len(negative_g))
    positive_g.extend(negative_g)
    return positive_g

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, device=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        at = torch.reshape(at, shape=targets.shape)
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        self.CE_RATIO = 0.8
        self.ALPHA = 0.8

    def forward(self, inputs, targets, smooth=1, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (self.ALPHA * ((targets * torch.log(inputs)) + ((1 - self.ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.CE_RATIO * weighted_ce) - ((1 - self.CE_RATIO) * dice)
        return combo

