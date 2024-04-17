import os
import sys 
import math
import torch
import pickle
import numpy as np
import setproctitle
from tqdm import tqdm
from rdkit import Chem
from sklearn import metrics
from datetime import datetime
from torchinfo import summary
from train import fitGraphNN_v2
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from train import saveSummaryTorchInfo
from torch_geometric.nn import GNNExplainer
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
sys.path.append("/home/scontino/python/graph_vae/")
from graphArchitecture import SingleClassGCNModel, SingleClassGCNModel_shapLike_2
from Package.Train_metrics.Metrics import MetricsClass
from Package.DataProcessing.DataProcessing import ManagePickle, DataProcessing


#/ Suppress annoying warnings
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


protein_name = "CDK1"

#/  Dataset creation
protein_list = ["ACK", "ALK", "CDK1", "CDK2", "CDK6", "INSR","ITK","JKA2","JNK3","MELK",
                "CHK1", "CK2A1", "CLK2", "DYRK1A", "EGFR","ERK2","GSK3B","IRAK4","MAP2k1","PDK1"]
protein_index = protein_list.index(protein_name)

setproctitle.setproctitle(f'single graphember ({protein_name})')


csv_path = "/home/psortino/classificatore/SMiles&LabelFull.csv"

base_dir = "/home/psortino/classificatore/"



print("Loading pickle dataset...")
with open(f"/home/scontino/python/graph_vae/Addestramento_SDF_07_23/pickleDataset/{protein_name}/{protein_name}_sdf_single_class.pickle", 'rb') as f:
    dataset = pickle.load(f)
    
graphs, smiles, labels = dataset

for i in tqdm(range(len(graphs))):
    graphs[i].y = torch.tensor(labels[i], dtype=torch.float32)


#/ path salvataggio 
n_prova = 111
path_models = f"/home/scontino/python/graph_vae/Addestramento_SDF_07_23/Esperimenti/{protein_name}/models/prova_{n_prova}/"
path_results = f"/home/scontino/python/graph_vae/Addestramento_SDF_07_23/Esperimenti/{protein_name}/results/prova_{n_prova}/"


os.makedirs(path_models, exist_ok=True)
os.makedirs(path_results, exist_ok=True)


#! model 
net = SingleClassGCNModel_shapLike_2(feature_node_dim=12, num_classes=1)
net = net.to(device)
saveSummaryTorchInfo(model=net, path_results=path_results)


def splitData(graphs, labels, split_threshold):
    pos_index = list(np.where(np.array(labels, dtype=np.uint8) == 1)[0])
    neg_index = list(np.where(np.array(labels, dtype=np.uint8) == 0)[0])

    graph_splitted_pos, graph_splitted_neg = [], []

    for idx_pos in pos_index:
        graph_splitted_pos.append(graphs[idx_pos])

    for idx_neg in neg_index[:len(pos_index) * split_threshold]:
        graph_splitted_neg.append(graphs[idx_neg])

    graph_splitted_pos.extend(graph_splitted_neg)
    return graph_splitted_pos


#/ splitto i dati 
labels = [g.y for g in graphs]
number_split = 50
graphs = splitData(graphs=graphs, labels=labels, split_threshold=number_split)
labels_splitted = [g.y for g in graphs]
x_tr, test_set = train_test_split(graphs, test_size=0.1, random_state=17, shuffle=True, stratify=labels_splitted)
labels_splitted = [g.y for g in x_tr]
train_set, val_set = train_test_split(x_tr, test_size=0.1, random_state=17, shuffle=True, stratify=labels_splitted)

#/ params 
b_size = 64
num_epochs = 2000
optim = torch.optim.Adamax(net.parameters(), lr=0.002)

#/ dataloader 
train_loader = DataLoader(train_set, batch_size=b_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_set, batch_size=b_size, drop_last=True, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(val_set), drop_last=True)

x = next(iter(test_loader))
print(sum(x.y))


#/ train 
train_loss, train_acc = fitGraphNN_v2(
    model=net, num_epochs=num_epochs,train_loader=train_loader, val_loader=val_loader, 
    device=device, optimizer=optim, classWeight=[1.0, 35.0],
    model_path=path_models, EarlyMonitor="val_loss", patience=50,
    EarlyStopping="Yes", fold=17, path_results=path_results, b_size=b_size, trainCheckpoint="combo",
    hook_start=False, hook_mid=False, hook_end=False
) 


#/ test 
#/ Calcolo le metriche 
tm = MetricsClass(input_shape=(7, 1024, 1))
dt = DataProcessing(Data="VectorLabel")
net2 = torch.load(path_models+"LossModel_17.pth").to(device)

pred2, labels_test = [], []
for data_test in tqdm(test_loader):
    data_test = data_test.to(device)
    outputs = net2(data_test.x[:, :12], data_test.edge_index, data_test.batch, isTrain=True).to(device)
    outputs = outputs.squeeze()
    pred2.extend(outputs.detach().cpu().numpy())
    labels_test.extend(data_test.y.detach().cpu().numpy())


#/ trasformo in numpy
pred2 = np.array(pred2, dtype=np.float64)
labels_test = np.array(labels_test, dtype=np.float64)


acc, los, sensitivity, zero_accuracy, MCC, roc_auc, f1, confusion, balanced, summaryRes = tm.metrics(yTrue=labels_test, yPred=pred2)
#/Salvo i risultati
tm.writeResults(summaryResults=summaryRes, confusion=confusion, resultsPath=path_results, tuningResultsPath=path_results, fileName=f"{protein_name}.csv", modelName="LossModel_17", combination=0, kinaseNumber="CDK1", parameter=[f"Rapporto 1:{number_split}"], fold=n_prova)
#/Creo la directory per l'enrichment factor
earlyDir = "EF/"
res_pat = path_results+earlyDir
if not os.path.exists(res_pat):
    os.makedirs(res_pat)

earlyDir2 = f"EF_LossModel_17/"
res_pat2 = res_pat+earlyDir2
if not os.path.exists(res_pat2):
    os.makedirs(res_pat2)
#/Lista percentuali 
earlyList = [0.01, 0.02, 0.05, 0.1] 
for xEF in earlyList:
    dim_test = confusion[0, 0] + confusion[0, 1] + confusion[1, 0] + confusion[1, 1]
    actiCompounds = confusion[1, 0] + confusion[1, 1]
    MCC_sk, percentActive, ActivePredict, activeCompounds2 = tm.EF_v2(resultsPath=res_pat2, yTest=labels_test, yPred=pred2, protein=f'CDK1', dimTestSet=dim_test, percent=float(xEF), fold=n_prova, activeCompounds=actiCompounds, fileName="EF_v2")
    MCC_sk_2, percentActive_2, ActivePredict_2, activeCompounds2_2 = tm.tpp(resultsPath=res_pat2, yTest=labels_test, yPred=pred2, protein=f'CDK1', dimTestSet=dim_test, percent=float(xEF), fold=n_prova, activeCompounds=actiCompounds, fileName="TPP")
