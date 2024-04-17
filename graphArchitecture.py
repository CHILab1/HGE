import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, GraphConv



class SingleClassGCNModel_shapLike_2(nn.Module):
    def __init__(self, feature_node_dim=9, num_classes=1):
        super(SingleClassGCNModel_shapLike_2, self).__init__()

        self.conv1 = None
        self.bn3 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(32)

        self.conv1 = GraphConv(feature_node_dim, 32, aggr="max")
        self.conv2 = GraphConv(32, 64, aggr="max")
        self.conv3 = GraphConv(64, 128, aggr="max")
        self.conv4 = GraphConv(128, 256, aggr="max")
        self.conv5 = GraphConv(256, 128, aggr="max")
        self.conv6 = GraphConv(128, 64, aggr="max")
        self.conv7 = GraphConv(64, 32, aggr="max")

        self.dropout = nn.Dropout(p=0.45)
        self.dropout2 = nn.Dropout(p=0.2)
        #? old one
        self.prelu = nn.PReLU()

        self.linear0 = nn.Linear(32, 32)
        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()
    
        self.final_conv_grads = []


    def activations_hook(self, grad):
        self.final_conv_grads.append(grad)
    
    def get_activations_gradient(self):
        return self.final_conv_grads
    
    
    def forward(self, x, edge_index, batch, hook_start:bool=False, hook_mid:bool=False, hook_end:bool=False):
        self.hook_start = hook_start
        self.hook_mid = hook_mid
        self.hook_end = hook_end
        #/ Graph convolutions with nonlinearity:
        self.conv_grad_1 = self.conv1(x, edge_index)
        if self.hook_start:
            self.conv_grad_1.register_hook(self.activations_hook)
        x = self.prelu(self.conv_grad_1)
        
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        self.conv_grad_2 = self.conv3(x, edge_index)
        if self.hook_mid:
            self.conv_grad_2.register_hook(self.activations_hook)
        x = self.prelu(self.bn3(self.conv_grad_2))
        x = self.conv4(x, edge_index)
        x = self.prelu(x)
        x = self.conv5(x, edge_index)
        x = self.prelu(x)
        x = self.conv6(x, edge_index)
        x = self.prelu(x)
        self.conv_grad_3 = self.conv7(x, edge_index)
        if self.hook_end:
            self.conv_grad_3.register_hook(self.activations_hook)
        x = self.prelu(self.bn5(self.conv_grad_3))

        
        #/ Graph embedding:
        x_embed = global_mean_pool(x, batch)

        #/ Linear classifier:
        x = self.linear0(x_embed)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.prelu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.prelu(x)
        x = self.linear3(x)
        y = self.sigmoid(x - 0.5) 
        return y, x