import numpy as np
import torch as t
from pyrwr.rwr import RWR
import random


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))

    return adj


def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def heteg(SR, SD, AM):
    reSR = np.hstack((SR, AM))
    reSD = np.hstack((AM.T ,SD))
    adj = np.vstack((reSR, reSD))
    return adj

def bipartite(SR, SD, AM):
    SR_matrix = np.matrix(np.zeros((SR.shape[0], SR.shape[0]), dtype=np.int8))
    SD_matrix = np.matrix(np.zeros((SD.shape[0], SD.shape[0]), dtype=np.int8))
    mat1 = np.hstack((SR_matrix, AM))
    mat2 = np.hstack((AM.T, SD_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = t.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = t.where(t.isinf(D_5), t.full_like(D_5, 0), D_5)
    L_D_11 = t.mm(D_5, L_D_1)
    L_D_11 = t.mm(L_D_11, D_5)
    return L_D_11


def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = t.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne


def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[t.nonzero(k, as_tuple=False)[0]]
    K[t.where(K == 0)] = min_v
    D = t.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S



def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    col_rand_array = np.arange(neg_index_matrix.shape[1])
    np.random.seed(2222)
    np.random.shuffle(col_rand_array)
    neg_index_matrix = neg_index_matrix[:, col_rand_array[0:pos_index_matrix.shape[1]]]
    neg_index = random_index(neg_index_matrix, sizes)
    pos_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index


class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.drug_size = drug_size
        self.mic_size = mic_size
        self.F1 = 128
        self.F2 = 64
        self.F3 = 32
        self.k_fold = 5
        self.epoch = 1000
        self.learn_rate = 0.0005
        self.seed = 1
        self.h1_gamma = 2 ** (-5)
        self.h2_gamma = 2 ** (-3)
        self.h3_gamma = 2 ** (-3)

        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-3)


def get_syn_sim(A, seq_sim, str_sim, mode):

    GIP_c_sim = GIP_kernel(A)
    GIP_d_sim = GIP_kernel(A.T)

    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_c = np.zeros((A.shape[0], A.shape[0]))
    syn_d = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_c[i, j] = GIP_c_sim[i, j]
            else:
                syn_c[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2


    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_d[i, j] = GIP_d_sim[i, j]
            else:
                syn_d[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_c, syn_d

def GIP_kernel(Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    # initate a matrix as results matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    # calculate the results matrix
    for i in range(nc):
        for j in range(nc):
            # calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def getGosiR(Asso_RNA_Dis):
    # calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r



def simabs(SIM):
    for i in range(SIM.shape[0]):
        for j in range(SIM.shape[0]):
            if SIM[i][j] < 0:
                SIM[i][j] = abs(SIM[i][j])
    return SIM

def formart_DISandRNA(RNA_CIS, Disease_DIS):
    DIS = []
    RNA = []
    for i in range(Disease_DIS.shape[0]):
        for j in range(Disease_DIS.shape[0]):
            DIS_inner = []
            DIS_inner.append(i)
            DIS_inner.append(j)
            DIS_inner.append(Disease_DIS[i][j])
            DIS.append(DIS_inner)
    for i in range(RNA_CIS.shape[0]):
        for j in range(RNA_CIS.shape[0]):
            RNA_inner = []
            RNA_inner.append(i)
            RNA_inner.append(j)
            RNA_inner.append(RNA_CIS[i][j])
            RNA.append(RNA_inner)
    np.savetxt('../data/rwr/RWR_FORMART_DIS.csv', DIS)
    np.savetxt('../data/rwr/RWR_FORMART_CIS.csv', RNA)
    return DIS, RNA

def SimtoRWR(RNA_CIS, Disease_DIS, FLAGS):
    RNA_CIS = simabs(RNA_CIS)
    Disease_DIS = simabs(Disease_DIS)
    path_DIS = '../data/rwr/RWR_FORMART_DIS.csv'
    path_CIS = '../data/rwr/RWR_FORMART_CIS.csv'
    formart_DISandRNA(RNA_CIS, Disease_DIS)
    RWRRNA = np.array(RWR_Comp(path_CIS, RNA_CIS.shape[0], 'RSim'))
    RWRDIS = np.array(RWR_Comp(path_DIS, Disease_DIS.shape[0], 'DSim'))
    return RWRRNA, RWRDIS

def RWR_Comp(path, w, FLAGS, NAME):
    FEATURE = []
    rwr = RWR()
    rwr.read_graph(path, "directed")
    for i in range(w):
        r = rwr.compute(i, c=0.3)
        FEATURE.append(sorted(r, reverse=True))
    print(NAME + ' finish')
    return FEATURE