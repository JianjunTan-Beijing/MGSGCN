# -*- coding:utf-8 -*-

import numpy as np
import networkx as nx


def construct_graph(data, nodes):
    # construct Bigraph
    if data.ndim == 1:
        data = data.reshape(-1,3)
    nodes_l = [i for i in list(set(data[:, 0]))]
    nodes_d = [i for i in list(set(data[:, 1]))]
    # nodes_l = [_l + 180 for _l in nodes_l]
    all_nodes_l, all_nodes_d = nodes['l'], nodes['d']
    print(len(nodes_l), len(nodes_d), len(all_nodes_l), len(all_nodes_d))
    Bigraph = nx.Graph()
    for line in data:
        node_l, node_d = line[0], line[1]
        Bigraph.add_node(node_l, bipartite=0)
        Bigraph.add_node(node_d, bipartite=1)
        Bigraph.add_edge(node_l, node_d, btype=line[2])
    # construct graph
    D_graph = dict()
    n_neigs_l = 0
    n_neigs_d = 0
    for u in all_nodes_l:
        if u in nodes_l:
            neighbors = Bigraph.edges(u)
            neigs_u = [i for u, i in neighbors]
            D_graph[u] = neigs_u
            n_neigs_l += len(neigs_u)
        else:
            D_graph[u] = []
    for i in all_nodes_d:
        if i in nodes_d:
            neighbors = Bigraph.edges(i)
            neigs_i = [u for i, u in neighbors]
            D_graph[i] = neigs_i
            n_neigs_d += len(neigs_i)
        else:
            D_graph[i] = []
    return D_graph



def construct_D_lncRNA_disease(data, nodes):
    graphs_t = dict()
    btypes = list(set(data[:, 2]))
    graph_base = construct_graph(data, nodes)    # 根据数据创建linRNA-疾病的二部图
    graphs_t['base'] = graph_base
    # 我们只要base图！
    # for btype in btypes:
    #     data_type = extract_edges_method(data, btype)
    #     graphs_t[str(btype)] = construct_graph(data_type, nodes)
    return graphs_t


def _generate_incidence_matrix(edges, n_smp):
    G = np.zeros((n_smp,n_smp))
    for key,val in edges.items():
        for v in val:
            G[v,key] = 1
    return G


def generate_incidence_matrix_multiple(graph):
    Gs = dict()
    btypes = graph.keys()
    n_smp = len(graph['base'])
    for btype in btypes:
        G = _generate_incidence_matrix(graph[btype], n_smp)
        Gs[btype] = G
    return Gs

def split_Gs(Gs, num_l):
    Gs_l,Gs_d = dict(),dict()  # 初始化两个空字典，存储lncRNA和疾病的图表示
    for key,val in Gs.items():
        Gs_l[key] = val[:num_l,num_l:]  # 这里拿到的不是本node的邻接矩阵，而是本node和另一种node的邻接矩阵
        Gs_d[key] = val[num_l:,:num_l]
    return Gs_l,Gs_d


def generate_G_from_BG(args, H):
    H = np.array(H) # 将H转换为numpy数组. H--邻接矩阵
    n_edge = H.shape[1] # 通过获取矩阵H中的列数计算图形中的边数
    W = np.ones(n_edge) # 初始化大小为n_edge的权重向量W，所有元素均设置为1 权重用于权衡图中每条边的贡献
    DV = np.sum(H * W, axis=1) # 通过元素H*W并沿行求和，计算节点度（每个节点上附带的权重之和）
    DE = np.sum(H, axis=0) # 沿H的列求和和计算边度（每条边的权重之和）
    # 为节点度和边度添加小数，避免除以0
    DV += 1e-12
    DE += 1e-12
    invDE = np.mat(np.diag(np.power(DE, -1))) # 用DE的倒数创建对角矩阵 DE^(-1)
    W = np.mat(np.diag(W)) # 创建对角矩阵W,边权重位于对角线上
    H = np.mat(H) # 将输入的H转换为numpy矩阵，用于矩阵乘法
    HT = H.T # 计算矩阵H的转置
    # 根据args.conv的值分支，表示使用的图卷积类型
    if args.conv == "sym":
        DV2 = np.mat(np.diag(np.power(DV, -0.5))) # 创建对角矩阵，其元素为DV的倒数平方根
        G = DV2 * H * W * invDE * HT * DV2   # 度归一化(DV2)、图邻接(H)、边缘加权(W)、边缘归一化(invDE)、H的转置(HT)和再次度归一化(DV2)
    elif args.conv == "asym":
        DV1 = np.mat(np.diag(np.power(DV, -1))) # 创建一个对角矩阵DV1，其中包含DV的元素倒数
        G = DV1 * H * W * invDE * HT

    return G  # 邻接矩阵归一化后结果。还未与init_embed相乘


def generate_Gs_from_O(args, DGs):  # DGs：邻接矩阵
    Gs = dict()
    for key,val in DGs.items():
        Gs[key] = generate_G_from_BG(args, val) # 返回D^(-1)*A*D^(-1)A.T，为计算图卷积，还应继续与XQ相乘，X-node emb, Q-参数
    return Gs