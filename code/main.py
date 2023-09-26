import numpy as np
import scipy.sparse as sp
import random
import gc
import pandas as pd
from sklearn.decomposition import PCA
import torch.nn as nn
from clac_metric import get_metrics
from util import *
import torch as t
from torch import optim
from loss import Myloss
from torch_geometric.data import Data

from model import MNGNN


def train(model, data_o, data_s, data_a, train_drug_mic_matrix,optimizer, sizes,index,drug_mic_matrix1,k):
    model.train()
    regression_crit = Myloss()
    # loss_fct = nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    lbl = data_a.y
    pre_matrix = np.zeros(drug_mic_matrix.shape)
    def train_epoch():
        # model.train()
        model.zero_grad()
        score, cla_os, cla_os_a, _= model(data_o, data_s, data_a)
        drug_mic_matrix = t.DoubleTensor(train_drug_mic_matrix)
        loss1 = regression_crit(drug_mic_matrix, score,model.drug_l, model.mic_l, model.alpha1,
                               model.alpha2, sizes)
        loss2 = b_xent(cla_os, lbl.float())
        loss3 = b_xent(cla_os_a, lbl.float())
        loss = loss1 + 0.1 * loss2 + 0.1 * loss3
        model.alpha1 = t.mm(
            t.mm((t.mm(model.drug_k, model.drug_k) + model.lambda1 * model.drug_l).inverse(), model.drug_k),
            2 * drug_mic_matrix - t.mm(model.alpha2.T, model.mic_k.T)).detach()
        model.alpha2 = t.mm(t.mm((t.mm(model.mic_k, model.mic_k) + model.lambda2 * model.mic_l).inverse(), model.mic_k),
                            2 * drug_mic_matrix.T - t.mm(model.alpha1.T, model.drug_k.T)).detach()
        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()
        model.eval()
        predict_y_proba = score.reshape(271, 218).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
        metric_tmp = get_metrics(drug_mic_matrix1[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])
        return loss

    for epoch in range(1, 500):
        train_reg_loss = train_epoch()

        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass


def PredictScore(train_drug_mic_matrix, drug_matrix, mic_matrix, seed, sizes,index,drug_mic_matrix,k):
    np.random.seed(seed)
    # train_data = {}
    # drug_matrix, mic_matrix = get_syn_sim(train_drug_mic_matrix, drug_matrix, mic_matrix, mode=1)
    # train_data['Y_train'] = t.DoubleTensor(train_drug_mic_matrix)
    # Heter_adj = constructHNet(train_drug_mic_matrix, drug_matrix, mic_matrix)
    # Heter_adj = t.FloatTensor(Heter_adj)
    # Heter_adj_edge_index = get_edge_index(Heter_adj)
    # train_data['Adj'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}
    #
    # X = constructNet(train_drug_mic_matrix)
    # X = t.FloatTensor(X)
    # train_data['feature'] = X

    adj1 = bipartite(drug_matrix, mic_matrix, train_drug_mic_matrix)
    # adj1 = adj1 + np.eye(adj1.shape[0])
    SR, SD = get_syn_sim(train_drug_mic_matrix, drug_matrix, mic_matrix, mode=1)
    SRS, SDS = SimtoRWR(SR, SD, 4)
    adj = sp.coo_matrix(adj1)
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj1 = t.FloatTensor(adj1)
    edge_index_o = get_edge_index(adj_o)
    #####高阶#########
    # adj_s = adj.dot(adj)
    # adj_s = adj_s.sign()
    adj_s = adj1**2
    adj_s = adj_s.sign()
    # adj2 = torch.FloatTensor(adj_s)
    edge_index_s = get_edge_index(adj_s)
    # features_o = adj_o.todense()
    # features_o = bipartite(SR, SD, train_drug_mic_matrix)
    features_o = heteg(SRS, SDS, train_drug_mic_matrix)
    # features_o = xiangsitezhe(SR, SD, AM)
    # pca = PCA(n_components=128)
    # pca.fit(features_o)
    # features_o = pca.transform(features_o)
    # adversarial nodes
    # np.random.seed(0)
    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]

    y_a = t.cat((t.ones(adj.shape[0], 1), t.zeros(adj.shape[0], 1)), dim=1)

    x_o = t.tensor(features_o, dtype=t.float)

    data_o = Data(x=x_o, edge_index=edge_index_o, adj=adj1)

    data_s = Data(edge_index=edge_index_s, adj=adj1)

    x_a = t.tensor(features_a, dtype=t.float)
    data_a = Data(x=x_a, y=y_a)

    model = MNGNN(aggregator="GCN", feature=489, hidden1=64, hidden2=32,
                  decoder1=128, dropout=0.6,sizes=sizes)


    # model = MKGCN.Model(sizes, drug_matrix, mic_matrix)
    # print(model)
    # for parameters in model.parameters():
    #     print(parameters, ':', parameters.size())

    optimizer = optim.Adam(model.parameters(), lr=sizes.learn_rate)

    # train(model, train_data, optimizer, sizes)
    train(model, data_o, data_s, data_a,train_drug_mic_matrix, optimizer, sizes,index,drug_mic_matrix,k)
    return model(data_o, data_s, data_a)





def cross_validation_experiment(drug_mic_matrix, drug_matrix, mic_matrix, sizes):
    index = crossval_index(drug_mic_matrix, sizes)
    metric = np.zeros((1, 7))
    pre_matrix = np.zeros(drug_mic_matrix.shape)
    print("seed=%d, evaluating drug-circrna...." % (sizes.seed))
    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_mic_matrix, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0
        drug_len = drug_mic_matrix.shape[0]
        dis_len = drug_mic_matrix.shape[1]
        drug_mic_res ,a,b,c= PredictScore(
            train_matrix, drug_matrix, mic_matrix, sizes.seed, sizes,index,drug_mic_matrix,k)
        predict_y_proba = drug_mic_res.reshape(drug_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
        metric_tmp = get_metrics(drug_mic_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / sizes.k_fold)
    metric = np.array(metric / sizes.k_fold)
    return metric, pre_matrix


if __name__ == "__main__":
    data_path = '../Data/MicrobeDrugA/'
    # data_set = 'aBiofilm/'
    data_set = 'MDAD/'
    # data_set = 'DrugVirus/'
    # drug_sim = np.loadtxt(data_path + data_set + 'drugsimilarity.txt', delimiter='\t')
    # mic_sim = np.loadtxt(data_path + data_set + 'microbesimilarity.txt', delimiter='\t')
    # adj_triple = np.loadtxt(data_path + data_set + 'adj.txt')
    # drug_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
    #                                 shape=(len(drug_sim), len(mic_sim))).toarray()
    drug_sim = pd.read_csv("../data/gene_seq_sim.csv", index_col=0).to_numpy()
    mic_sim = pd.read_csv("../data/drug_str_sim.csv", index_col=0).to_numpy()
    drug_mic_matrix = pd.read_csv("../data/association.csv", index_col=0).to_numpy()

    average_result = np.zeros((1, 7), float)
    circle_time = 1
    sizes = Sizes(drug_sim.shape[0], mic_sim.shape[0])
    results = []

    result, pre_matrix = cross_validation_experiment(
        drug_mic_matrix, drug_sim, mic_sim, sizes)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])