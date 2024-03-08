from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pandas as pd
import random

def load_new_feature_data(path_dataset, disease_feature_name, lnc_rna_feature_name):
    net = pd.read_excel(f'{path_dataset}/LDA.xlsx', index_col=0).values  # (894,280) 互作矩阵
    #输入lncRNA、疾病特征矩阵
    df_feats_L = pd.read_excel(f'{path_dataset}/{lnc_rna_feature_name}', index_col=None, header=None) # lnc-rna feat (894, 894)
    df_feats_D = pd.read_excel(f'{path_dataset}/{disease_feature_name}', index_col=None, header=None) # dis-feat (280, 280)

    u_features = df_feats_L.values
    v_features = df_feats_D.values
    return net, u_features, v_features


def load_Dataset2_data():
    path_dataset = 'Datasets/Dataset2'
    net = pd.read_excel(f'{path_dataset}/LDA.xlsx', index_col=0).values  # (894,280) 互作矩阵
    #输入lncRNA、疾病特征矩阵
    df_feats_L = pd.read_excel(f'{path_dataset}/LFGC_average.xlsx', index_col=None, header=None) # lnc-rna feat (894, 894)
    df_feats_D = pd.read_excel(f'{path_dataset}/DSGC_average.xlsx', index_col=None, header=None) # dis-feat (280, 280)

    u_features = df_feats_L.values
    v_features = df_feats_D.values
    return net, u_features, v_features


def load_Dataset1_data():
    path_dataset = 'Datasets/Dataset1'
    net = pd.read_excel(f'{path_dataset}/LDA.xlsx', index_col=0).values  # (180,59) 互作矩阵
    # 输入lncRNA、疾病特征矩阵
    df_feats_L = pd.read_excel(f'{path_dataset}/LFGC_average.xlsx', index_col=None, header=None)  # lnc-rna feat (180, 180)
    df_feats_D = pd.read_excel(f'{path_dataset}/DSGC_average.xlsx', index_col=None, header=None)  # dis-feat (59, 59)

    u_features = df_feats_L.values
    v_features = df_feats_D.values
    return net, u_features, v_features


def load_data(dataset, disease_feature_name, lnc_rna_feature_name):
    print("Loading lncRNAdisease dataset")
    if dataset in ['Dataset1', 'Dataset2']:
        net, u_features, v_features = load_new_feature_data(path_dataset = 'Datasets/%s'% dataset, disease_feature_name=disease_feature_name, lnc_rna_feature_name=lnc_rna_feature_name)
    else:
        path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat' #training_test_dataset.mat
        data=scio.loadmat(path_dataset)
        net=data['interMatrix']  # (285,226) 互作矩阵

        #lncRNA features and disease features
        u_features=data['lncSim'] # (285, 285) rna-feat
        disSim_path='raw_data/' + dataset + '/disSim.xlsx'
        disSim_data=pd.read_excel(disSim_path,header=0)
        v_features=np.array(disSim_data)  # (226,226) dis-feat

    num_list=[len(u_features)]
    num_list.append(len(v_features))
    u_features=np.hstack((u_features,net)) # 将互作矩阵和其相关性矩阵hstack，都作为rna-feature
    v_features=np.hstack((net.T,v_features))

    a=np.zeros((1,u_features.shape[0]+v_features.shape[0]),int)
    b=np.zeros((1,v_features.shape[0]+u_features.shape[0]),int)
    u_features=np.vstack((a,u_features))  # 各加了一行全0在第一行
    v_features=np.vstack((b,v_features))

    num_lncRNAs=net.shape[0]
    num_diseases=net.shape[1]

    row,col,_=sp.find(net)
    perm=random.sample(range(len(row)),len(row)) # 随机打乱顺序
    row,col=row[perm],col[perm]
    sample_pos=(row,col)
    print("the number of all positive sample:",len(sample_pos[0]))

    print("sampling negative links for train and test")
    X=np.ones((num_lncRNAs,num_diseases))
    net_neg=X-net
    row_neg,col_neg,_=sp.find(net_neg)
    perm_neg=random.sample(range(len(row_neg)), len(row))   # 采样和正样本一样多的负样本对
    row_neg,col_neg=row_neg[perm_neg],col_neg[perm_neg]
    sample_neg=(row_neg,col_neg)
    sample_neg=list(sample_neg)
    print("the number of all negative sample:", len(sample_neg[0]))

    u_idx = np.hstack([sample_pos[0], sample_neg[0]])
    v_idx = np.hstack([sample_pos[1], sample_neg[1]])
    labels= np.hstack([[1]*len(sample_pos[0]), [0]*len(sample_neg[0])])
    return u_features, v_features, net, labels, u_idx, v_idx, num_list