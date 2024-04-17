import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from contextlib import redirect_stdout
from utilities import Weighted_BCE_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR


def modelCheckpoint(model, epoch, model_path, optimizer, loss_training, loss_val, acc_val, fileName):
    """[summary]

    Args:
        model (model): pytorch model to save
        epoch (int): epochs
        model_path (str): path where save model in format "folder/folder/"
        optimizer (--): optimizer used during training
        loss_training (flaot): value of training loss
        loss_val (flaot): value of validation loss
        acc_val (flaot): value of validation accuracy
        fileName (str): name of model
    """
    fileName_weight = f"{fileName}.tar"
    fileName_model = f"{fileName}.pth"
    savePath_name_w = model_path+fileName_weight
    savePath_name_m = model_path+fileName_model
    #/salvo i pesi e tutte le info di addestramento
    torch.save({'epoch': epoch, 
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_validation': loss_val,
    'accuracy_validation':acc_val,
    'loss_training': loss_training
    }, savePath_name_w)
    #/ salvo l'intero modello 
    torch.save(model, savePath_name_m)

def saveLossHistory(epoch, loss_value, path_results, train=True):
    if train:
        file = open(os.path.join(path_results, f"train_loss.csv"), "a")
        if epoch == 0:
            file.writelines("Epoch,Train_loss\n")
            file.writelines(f"{epoch},{loss_value}\n")
        else:
            file.writelines(f"{epoch},{loss_value}\n")
        file.close()
    else:
        file = open(os.path.join(path_results, f"val_loss.csv"), "a")
        if epoch == 0:
            file.writelines("Epoch,Val_loss\n")
            file.writelines(f"{epoch},{loss_value}\n")
        else:
            file.writelines(f"{epoch},{loss_value}\n")
        file.close()
    
def fitGraphNN_v2(model, num_epochs, train_loader, val_loader, device, optimizer, model_path, path_results, classWeight, EarlyMonitor="val_loss", patience=50, trainCheckpoint="combo", EarlyStopping="Yes", fold=17, b_size=64, hook_start:bool=False, hook_mid:bool=False, hook_end:bool=False):
    """
    Train validation 

    # Args:
     * model (--):Pytorch Architecture 
     * num_epochs (int): number of training epochs
     * train_x (array): x-train array
     * train_y (array): y_train array
     * val_x (array): x-validation array
     * val_y (array): y-validation array
     * batch_size (int): batch size
     * JsonParameter ([type]): Json file with the * information of architecture and input
     * device (--): pytorch.device
     * optimizer (--): pytorch optimizer used
     * loss_function (str, optional): Loss * function. Defaults to "binary_crossentropy".
     * EarlyMonitor (str, optional): Monitor of * Early Stopping. Defaults to "val_loss".
     * patience (int, optional): Patience of Early * Stopping. Defaults to 50.
     * EarlyStopping (str, optional): Early * Stopping setting. Defaults to "Y" * (optional: "Yes", "yes")
     * trainCheckpoint (str, optional): * modelCheckpoint. Defaults to "None" * (optional: "val_loss", "val_acc" or "combo" * if you want to save 2 models. Best val_acc * model and best val_loss model)
     * classWeight (str, optional): * a Tensor of different weight of imbalanced class 
     
    # Raises:
        - ValueError: if num_epochs is less than patience a valueError is raised 

    # Returns:
     * Training_loss, validation_loss (list): A list of training loss and a list of validation loss is returned
    """

    num_epochs = num_epochs
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    epochs_no_improve = 0
    if EarlyStopping == "Y" or EarlyStopping == "yes" or EarlyStopping == "Yes":
        if patience > num_epochs:
            raise ValueError("num epochs must be higher than patience")
        for epoch in range(num_epochs):
            print(f'Epochs n:{epoch+1}/{num_epochs}')
            batch_len_train = 0
            batch_len_val = 0
            val_acc = 0
            train_loss_batch, train_acc_batch = [], []
            #/ Dico al modello che stiamo facendo train
            model.train()
           
            # scheduler = CosineAnnealingLR(optimizer,
            #                   T_max = 100, # Maximum number of iterations.
            #                  eta_min = 1e-4) # Minimum learning rate.
            
            for data in tqdm(train_loader):
                data = data.to(device)
                outputs = model(x=data.x[:, :12], edge_index=data.edge_index, batch=data.batch, hook_start=hook_start, hook_mid=hook_mid, hook_end=hook_end)
                outputs = outputs[0].squeeze()
                #/ select a loss function 
                # loss = nn.functional.binary_cross_entropy(outputs, data.y, weight=classWeight)
                #! weighted BCE Paolo
                b = Weighted_BCE_loss(pos_weight=classWeight[1], neg_weight=classWeight[0])
                loss = b(y_pred=outputs, y_true=data.y)
                #/ Backward and optimize
                optim = optimizer
                optim.zero_grad()
                loss.backward() #/ esegue la backpropagation per noi
                optim.step() #/ ottimizza i pesi ad ogni step 
              
                train_loss_batch.append(loss.item())
                torch.cuda.empty_cache()
                
                #/ accuracy train 
                acc = accuracy_score(data.y.detach().cpu().numpy(), np.round(outputs.detach().cpu().numpy()))
                train_acc_batch.append(acc)
            #/ print accuracy e loss train 
            # scheduler.step()
            print(f"Training Loss:\t{sum(train_loss_batch)/len(train_loss_batch)}\nTraining Accuracy:{sum(train_acc_batch)/len(train_acc_batch)}")
            train_loss.append(sum(train_loss_batch)/len(train_loss_batch))
            train_acc.append(sum(train_acc_batch)/len(train_acc_batch))
            
            saveLossHistory(epoch=epoch, loss_value=sum(train_loss_batch)/len(train_loss_batch), path_results=path_results)
            
            del train_loss_batch, train_acc_batch
            
            #/ validation step 
            print("############################################")
            print("\nValidation step\n")
            print("############################################")
            #/ Dico al modello che stiamo facendo validation
            model.eval()
            with torch.no_grad():
                valid_loss_batch, valid_acc_batch = [], []
                for data_val in tqdm(val_loader):
                    data_val = data_val.to(device)                    
                    #/ validation step 
                    val_out = model(x=data_val.x[:, :12], edge_index=data_val.edge_index, batch=data_val.batch, hook_start=hook_start, hook_mid=hook_mid, hook_end=hook_end)
                    val_out = val_out[0].squeeze()
                    # b = nn.functional.binary_cross_entropy(val_out, data_val.y, weight=classWeight)
                    # valid_loss_batch.append(b.item())
                    #! weighted_BCE_paolo
                    b = Weighted_BCE_loss(pos_weight=classWeight[1], neg_weight=classWeight[0])
                    loss = b(y_pred=val_out, y_true=data_val.y)
                    valid_loss_batch.append(loss)
                    #/ val accuracy calculation
                    acc = accuracy_score(data_val.y.detach().cpu().numpy(), np.round(val_out.detach().cpu().numpy()))
                    valid_acc_batch.append(acc)
                saveLossHistory(epoch=epoch, loss_value=sum(valid_loss_batch)/len(valid_loss_batch), path_results=path_results, train=False)
                
                torch.cuda.empty_cache()
                #/ early stopping 
                valid_loss = sum(valid_loss_batch)/len(valid_loss_batch)
                valid_acc = sum(valid_acc_batch)/len(valid_acc_batch)
                
                if epoch == 0:
                    threshold_loss_valid_batch = valid_loss
                    threshold_acc_valid_batch = 0
                    print(f"La val_loss é:\t{threshold_loss_valid_batch} -- val_acc: {threshold_acc_valid_batch}")
                    modelCheckpoint(model=model, epoch=epoch, model_path=model_path, optimizer=optimizer, loss_training=train_loss, loss_val=threshold_loss_valid_batch, acc_val=valid_acc, fileName=f"LossModel_{fold}")
                    pass
                elif epoch != 0:
                    if valid_loss < threshold_loss_valid_batch:
                        print(f"La val_loss è minore della soglia:\t{threshold_loss_valid_batch}")
                        print(f"La nuova val_loss è la seguente:\t{valid_loss}")
                        threshold_loss_valid_batch = valid_loss
                        epochs_no_improve = 0
                        modelCheckpoint(model=model, epoch=epoch, model_path=model_path, optimizer=optimizer, loss_training=train_loss, loss_val=threshold_loss_valid_batch, acc_val=valid_acc, fileName=f"LossModel_{fold}")
                        if valid_acc > threshold_acc_valid_batch:
                            threshold_acc_valid_batch = valid_acc
                            print(f"La val_acc è superiore alla soglia:\t{threshold_acc_valid_batch}")
                            print(f"La nuova val_acc è la seguente:\t{valid_acc}")
                            if trainCheckpoint == "val_acc" or trainCheckpoint == "combo":
                                modelCheckpoint(model=model, epoch=epoch, model_path=model_path, optimizer=optimizer, loss_training=train_loss, loss_val=threshold_acc_valid_batch, acc_val=valid_acc, fileName=f"AccModel_{fold}")
                        else:
                            print(f"La val_acc non è migliorata:\t{threshold_acc_valid_batch}")
                    else:
                        print(f"La val_loss non è diminuita rispetto a:\t{threshold_loss_valid_batch}")
                        epochs_no_improve += 1
                        print(f"La loss non diminuisce da:\t{epochs_no_improve}")
                if epochs_no_improve == patience: 
                    print('Early stopping!')
                    early_stop = True
                    print("Stopped")
                    break
                else:
                    continue
                
        return train_loss, train_acc 
    
    else:
        print("Devi usare l'Early stopping")

def saveSummaryTorchInfo(model, path_results):
    with open(path_results+"ModelSummary.txt", 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            summary(model)