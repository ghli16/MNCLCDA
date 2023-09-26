import torch as t
# from torch import nn
# from torch_geometric.nn import conv
from util import *


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch.nn.parameter import Parameter
from torch_geometric.nn import conv


# class Model(nn.Module):
#     def __init__(self, sizes, drug_sim, mic_sim):
#         super(Model, self).__init__()
#         np.random.seed(sizes.seed)
#         t.manual_seed(sizes.seed)
#         self.drug_size = sizes.drug_size
#         self.mic_size = sizes.mic_size
#         self.F1 = sizes.F1
#         self.F2 = sizes.F2
#         self.F3 = sizes.F3
#         self.seed = sizes.seed
#         self.h1_gamma = sizes.h1_gamma
#         self.h2_gamma = sizes.h2_gamma
#         self.h3_gamma = sizes.h3_gamma
#
#         self.lambda1 = sizes.lambda1
#         self.lambda2 = sizes.lambda2
#
#         self.kernel_len = 4
#         self.drug_ps = t.ones(self.kernel_len) / self.kernel_len
#         self.mic_ps = t.ones(self.kernel_len) / self.kernel_len
#
#         self.drug_sim = t.DoubleTensor(drug_sim)
#         self.mic_sim = t.DoubleTensor(mic_sim)
#
#         self.gcn_1 = conv.GCNConv(self.drug_size + self.mic_size, self.F1)
#         self.gcn_2 = conv.GCNConv(self.F1, self.F2)
#         self.gcn_3 = conv.GCNConv(self.F2, self.F3)
#
#         self.alpha1 = t.randn(self.drug_size, self.mic_size).double()
#         self.alpha2 = t.randn(self.mic_size, self.drug_size).double()
#
#         self.drug_l = []
#         self.mic_l = []
#
#         self.drug_k = []
#         self.mic_k = []
#
#     def forward(self, input):
#         t.manual_seed(self.seed)
#         x = input['feature']
#         adj = input['Adj']
#         drugs_kernels = []
#         mic_kernels = []
#         H1 = t.relu(self.gcn_1(x, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
#         drugs_kernels.append(t.DoubleTensor(getGipKernel(H1[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
#         mic_kernels.append(t.DoubleTensor(getGipKernel(H1[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))
#
#         H2 = t.relu(self.gcn_2(H1, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
#         drugs_kernels.append(t.DoubleTensor(getGipKernel(H2[:self.drug_size].clone(), 0, self.h2_gamma, True).double()))
#         mic_kernels.append(t.DoubleTensor(getGipKernel(H2[self.drug_size:].clone(), 0, self.h2_gamma, True).double()))
#
#         H3 = t.relu(self.gcn_3(H2, adj['edge_index'], adj['data'][adj['edge_index'][0], adj['edge_index'][1]]))
#         drugs_kernels.append(t.DoubleTensor(getGipKernel(H3[:self.drug_size].clone(), 0, self.h3_gamma, True).double()))
#         mic_kernels.append(t.DoubleTensor(getGipKernel(H3[self.drug_size:].clone(), 0, self.h3_gamma, True).double()))
#
#         drugs_kernels.append(self.drug_sim)
#         mic_kernels.append(self.mic_sim)
#
#         drug_k = sum([self.drug_ps[i] * drugs_kernels[i] for i in range(len(self.drug_ps))])     ##### 271*271
#         self.drug_k = normalized_kernel(drug_k)
#         mic_k = sum([self.mic_ps[i] * mic_kernels[i] for i in range(len(self.mic_ps))])
#         self.mic_k = normalized_kernel(mic_k)
#         self.drug_l = laplacian(drug_k)
#         self.mic_l = laplacian(mic_k)
#
#         out1 = t.mm(self.drug_k, self.alpha1)
#         out2 = t.mm(self.mic_k, self.alpha2)
#
#         out = (out1 + out2.T) / 2
#
#         return out


class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        # self.activation = ng.Activation('sigmoid')  # 定义sigmoid激活函数
        # 获取维度为(embedding_size, embedding_size)的参数矩阵，即论文中的Q参数矩阵

        self.W = Parameter(torch.randn(feature_size, feature_size))

    def forward(self, h_diseases, h_mirnas):
        h_diseases0 = torch.mm(h_diseases, self.W)
        h_mirnas0 = torch.mul(h_diseases0, h_mirnas)
        # h_mirnas0 = h_mirnas.tanspose(0,1)
        # h_mirnsa0 = torch.mm(h_diseases0, h_mirnas0)
        h0 = h_mirnas0.sum(1)
        h = torch.sigmoid(h0)
        # h = h.unsequence(1)
        return h


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, h_diseases, h_mirnas):
        x = torch.mul(h_diseases, h_mirnas).sum(1)
        x = torch.reshape(x, [-1])
        outputs = F.sigmoid(x)
        return outputs








def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class MNGNN(nn.Module):
    def __init__(self, aggregator, feature, hidden1, hidden2, decoder1, dropout,sizes):
        super(MNGNN, self).__init__()
        np.random.seed(sizes.seed)
        t.manual_seed(sizes.seed)
        self.drug_size = 271
        self.mic_size = 218
        self.h1_gamma = sizes.h1_gamma
        self.alpha1 = t.randn(self.drug_size, 218).double()
        self.alpha2 = t.randn(self.mic_size, 271).double()
        self.lambda1 = sizes.lambda1
        self.lambda2 = sizes.lambda2

        self.drug_l = []
        self.mic_l = []

        self.drug_k = []
        self.mic_k = []
        if aggregator == 'GIN':
            self.mlp_o1 = MLP(feature, hidden1)
            self.mlp_o2 = MLP(hidden1 * 2, hidden2)
            self.mlp_s1 = MLP(feature, hidden1)
            self.mlp_s2 = MLP(hidden1 * 2, hidden2)

            self.encoder_o1 = GINConv(self.mlp_o1, train_eps=True).jittable()
            self.encoder_o2 = GINConv(self.mlp_o2, train_eps=True).jittable()
            self.encoder_s1 = GINConv(self.mlp_s1, train_eps=True).jittable()
            self.encoder_s2 = GINConv(self.mlp_s2, train_eps=True).jittable()

        elif aggregator == 'GCN':
            # self.encoder_o1 = GCNConv(feature, hidden1)
            # self.encoder_o2 = GCNConv(hidden1 * 2, hidden2)
            # self.encoder_s1 = GCNConv(feature, hidden1)
            # self.encoder_s2 = GCNConv(hidden1 * 2, hidden2)

            self.encoder_o1 = conv.GCNConv(feature, hidden1)
            self.encoder_o2 = conv.GCNConv(hidden1 * 2, hidden2)
            self.encoder_s1 = conv.GCNConv(feature, hidden1)
            self.encoder_s2 = conv.GCNConv(hidden1 * 2, hidden2)


        # self.decoder1 = nn.Linear(hidden2 * 2 * 4, 1)
        # self.decoder2 = nn.Linear(decoder1, 32)
        # self.decoder3 = nn.Linear(32, 1)
        self.decoder = BilinearDecoder(hidden1)
        # self.decoder = InnerProductDecoder()




        self.disc = Discriminator(hidden2 * 2)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data_o, data_s, data_a):

        x_o, adj = data_o.x, data_o.edge_index
        adj2 = data_s.edge_index
        x_a = data_a.x
        # x_o, adj,ADJ = data_o.x, data_o.edge_index,data_o.adj
        # adj2 ,ADJ2= data_s.edge_index,data_s.adj
        # x_a = data_a.x



        x1_o = F.relu(self.encoder_o1(x_o, adj))
        # x1_o = F.relu(self.encoder_o1(x_o, adj,ADJ[adj[0],adj[1]]))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_s = F.relu(self.encoder_s1(x_o, adj2))
        # x1_s = F.relu(self.encoder_s1(x_o, adj2,ADJ2[adj[0],adj[1]]))
        x1_s = F.dropout(x1_s, self.dropout, training=self.training)

        x1_os = torch.cat((x1_o, x1_s), dim=1)

        x2_o = self.encoder_o2(x1_os, adj)
        x2_s = self.encoder_s2(x1_os, adj2)

        # x2_o = self.encoder_o2(x1_os, adj,ADJ[adj[0],adj[1]])
        # x2_s = self.encoder_s2(x1_os, adj2,ADJ2[adj[0],adj[1]])
        x2_os = torch.cat((x2_o, x2_s), dim=1)

        x1_o_a = F.relu(self.encoder_o1(x_a, adj))
        # x1_o_a = F.relu(self.encoder_o1(x_a, adj,ADJ[adj[0],adj[1]]))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x1_s_a = F.relu(self.encoder_s1(x_a, adj2))
        # x1_s_a = F.relu(self.encoder_s1(x_a, adj2,ADJ2[adj[0],adj[1]]))
        x1_s_a = F.dropout(x1_s_a, self.dropout, training=self.training)

        x1_os_a = torch.cat((x1_o_a, x1_s_a), dim=1)

        x2_o_a = self.encoder_o2(x1_os_a, adj)
        x2_s_a = self.encoder_s2(x1_os_a, adj2)
        # x2_o_a = self.encoder_o2(x1_os_a, adj,ADJ[adj[0],adj[1]])
        # x2_s_a = self.encoder_s2(x1_os_a, adj2,ADJ2[adj[0],adj[1]])

        x2_os_a = torch.cat((x2_o_a, x2_s_a), dim=1)

        # graph representation

        h_os = self.read(x2_os)
        h_os = self.sigm(h_os)
        h_os_a = self.read(x2_os_a)
        h_os_a = self.sigm(h_os_a)


        ret_os = self.disc(h_os, x2_os, x2_os_a)
        ret_os_a = self.disc(h_os_a, x2_os_a, x2_os)


        # x2_os = torch.cat((f, x2_os), dim=1)
        x2_osrna = x2_os[:271, :]
        x2_osdrug =x2_os[271:, :]

        drug_k = t.DoubleTensor(getGipKernel(x2_os[:271].clone(), 0, self.h1_gamma, True).double())
        self.drug_k = normalized_kernel(drug_k)
        mic_k = t.DoubleTensor(getGipKernel(x2_os[271:].clone(), 0, self.h1_gamma, True).double())
        self.mic_k = normalized_kernel(mic_k)
        self.drug_l = laplacian(drug_k)
        self.mic_l = laplacian(mic_k)
        out1 = t.mm(self.drug_k, self.alpha1)
        out2 = t.mm(self.mic_k, self.alpha2)
        out = (out1 + out2.T) / 2
        # entity1 = x2_osrna[src]     #### (256,64)
        # entity2 = x2_osdrug[dst]     #### (256,64)

        # add = entity1 + entity2     ####（256，64）
        # product = entity1 * entity2 ####（256，64）
        # concatenate = torch.cat((entity1, entity2), dim=1)   ##### (256,128)
        #
        # feature = torch.cat((add, product, concatenate), dim=1)  #### (256,256)

        # decoder
        # log = self.decoder(entity1,entity2)
        # log = self.decoder2(log)
        # log = self.decoder3(log)

        # self.drug_l = laplacian(x2_osrna)
        # self.mic_l = laplacian(x2_osdrug)

        #
        # out1 = t.mm(x2_osrna, self.alpha1)
        # out2 = t.mm(x2_osdrug, self.alpha2)
        # #
        # out = (out1 + out2.T) / 2
        #
        #         return out


        return out, ret_os, ret_os_a, x2_os