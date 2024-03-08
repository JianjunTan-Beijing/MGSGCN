import torch
import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing,GATConv
import torch.nn.functional as F
from torch.nn import Linear, Conv1d,AdaptiveMaxPool2d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool, global_max_pool,global_mean_pool
from torch_geometric.utils import dropout_adj
from util_functions import *
from torch.nn.parameter import Parameter



class MGSGCN(torch.nn.Module):
    # The MGSGCN model use GCN layer + GAT layer
    def __init__(self, in_features, gconv=GATConv, latent_dim=[16, 16, 16, 16], side_features=False, n_side_features=0, node_dims=(0, 0)):
        super(MGSGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.conv1 = GCNConv(in_features, 16)
        self.convs.append(gconv(16, latent_dim[0],heads=16,dropout=0.2))
        self.convs.append(gconv(16*16, 16,heads=16,dropout=0.2))
        self.convs.append(gconv(16*16, 16,heads=16,dropout=0.2))
        #self.convs.append(gconv(16*4, 2,heads=4,dropout=0.2))
        self.conv2=gconv(in_channels=2*16,out_channels=2,dropout=0.2,heads=16,concat=True)
        #self.lin1 = Linear(3*sum(latent_dim), 8)
        #self.lin2 = Linear(2*4*16, 8)

        self.weight_d = Parameter(torch.zeros(n_side_features, node_dims[1]))
        nn.init.xavier_uniform_(self.weight_d, gain=1.414)  # xavier初始化
        self.weight_right = Parameter(torch.Tensor(node_dims[0], self.conv1.out_channels))
        nn.init.xavier_uniform_(self.weight_right, gain=1.414)

        self.weight_dnn = Parameter(torch.Tensor(256 * 2, 2))
        #self.weight_dnn = Parameter(torch.Tensor(self.conv1.out_channels * 2, 2))
        nn.init.xavier_uniform_(self.weight_dnn, gain=1.414)  # xavier初始化

    def forward(self, data, H_l, H_d, need_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states=[]
        H_l_base, H_d_base = H_l['base'], H_d['base']
        conv1 = self.conv1(x, edge_index)  # (727, 16)

        '''交互信息传播：
        XD = XD.matmul(self.weight_d)
        HiT_D = torch.transpose(HD, 0, 1)
        X = X + HiT_D.matmul(XD)
        '''  # 这些简单来说就是H_P * X_p * S_p，其中H_P:交互信息矩阵, X_p: node-emd, self.weight_p: learnable coefficient
        XD = x.matmul(self.weight_d) # (727, 226)
        HiT_D = torch.transpose(H_d_base, 0, 1)  # (285, 226)
        XD_W = HiT_D.matmul(XD.T) # (285, 727)
        dual_conv = XD_W.T.matmul(self.weight_right) # (727,16)

        x = F.elu(conv1 + dual_conv)  # dim缩放 in_features->16

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        fin_layer_vec = torch.cat([x[users], x[items]], 1) # data.x.shape=8(814, 515), x.shape=(814, 2)
        out = fin_layer_vec.matmul(self.weight_dnn)
        out = F.log_softmax(out, dim=1)
        if need_embedding:
            vec = torch.cat([concat_states[users], concat_states[items]], 1)
            return out, vec  # todo: 应该是x？
        else:
            return out

    def predict(self, data, H_l, H_d):
        out=self.forward(data, H_l['base'], H_d['base'])
        return out