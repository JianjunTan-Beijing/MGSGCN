# -*- coding: utf-8 -*-
import ipdb
import gc
from preprocessing_Dataset import *
from train import *
from models import *
import argparse
import torch
from torch.utils.data import IterableDataset
import numpy as np
fontsize = 10  # 字号
class TestGraphDataset(IterableDataset):
    def __init__(self, u_indices, v_indices, all_indices, net, hop, u_features, v_features, num_batch=10):
        self.u_indices = u_indices
        self.v_indices = v_indices
        self.all_indices = all_indices
        self.net = net
        self.hop = hop
        self.u_features = u_features
        self.v_features = v_features
        self.num_batch = num_batch

    def __iter__(self):
        # make real_test_graph
        print("构建全部test-graphs")
        M, N = len(set(self.u_indices)), len(set(self.v_indices))
        edge_index_all = torch.cartesian_prod(torch.arange(M), torch.arange(N)).t().contiguous()
        positive_pairs = np.array(self.all_indices)[:, :621].T.tolist()
        edge_labels_all = torch.zeros_like(torch.Tensor(self.net))
        for u_idx, v_idx in positive_pairs:
            edge_labels_all[u_idx, v_idx] = 1.

        # 分批yield数据出来进行predict
        edge_labels_all_flat = edge_labels_all.flatten()

        # 计算每个批次的实际大小
        total_elements = len(edge_labels_all_flat)
        _bz = (total_elements + self.num_batch - 1) // self.num_batch  # 确保所有数据都被包含

        for batch_num in range(self.num_batch):
            start_index = batch_num * _bz
            end_index = min((batch_num + 1) * _bz, total_elements)  # 确保最后一个批次不会超出总元素数

            one_batch = edge_labels_all_flat[start_index:end_index]
            edge_index_all_batch = edge_index_all[:, start_index:end_index]
            final_test_graph = self.extracting_subgraphs(self.net, edge_index_all_batch, one_batch, self.hop,
                                                         self.u_features, self.v_features, self.hop * 2 + 1)
            yield final_test_graph


    def extracting_subgraphs(self, A, all_indices, all_labels, h=1, u_features=None, v_features=None, max_node_label=None):
        if max_node_label is None:  # if not provided, infer from graphs
            max_n_label = {'max_node_label': 0}
        class_values=np.array([0,1],dtype=float)
        def helper(A, links, g_labels):
            g_list = []
            ind=[[],[]]
            results = []
            for i, j, g_label in tzip(links[0], links[1], g_labels):
                _r = parallel_worker(g_label, (i, j), A, h, u_features, v_features, class_values)
                results.append(_r)
            # ==========================================================
            for i in range(len(results)):
                ind[0].append(results[i][4][0])
                ind[1].append(results[i][4][1])

            g_list += [
                nx_to_PyGGraph(g, g_label, n_labels, n_features, max_node_label, class_values)
                for g_label, g, n_labels, n_features, ind in tqdm(results)]

            del results
            return g_list

        graphs = helper(A, all_indices, all_labels)
        return graphs


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action='store_true', default=False)
    parser.add_argument('--dataset', help='dataset name', default='Dataset2')  # Dataset1, Dataset2 数据集
    parser.add_argument('--feature_name', help='feature name', default='DSS+DGS+DCS+LFS+LGS+LCS')  # 特征名字
    parser.add_argument('--use-features', action='store_true', default=True)
    parser.add_argument('--split-num', default=20)
    args = parser.parse_args()

    dataset = args.dataset
    feature_name = args.feature_name
    print("dataset:%s, feature_name:%s" % (dataset, feature_name))
    feature_names_split = feature_name.split("+LFS")
    lnc_rna_feature_name = "LFS.xlsx"
    if len(feature_names_split) > 1:
        lnc_rna_feature_name = "LFS%s.xlsx" % feature_names_split[1]
    disease_feature_name = "%s.xlsx" % feature_names_split[0]
    print("disease_feature_name:%s, lnc_rna_feature_name:%s" % (disease_feature_name, lnc_rna_feature_name))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2341)
    seed = 2341
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hop = 1

    if not args.no_train:
        # Construct model
        print('training.....')
        data_combo = (args.dataset, '', '')
        u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset, disease_feature_name, lnc_rna_feature_name)
        print('preprocessing end.')
        adj = torch.tensor(net)
        if args.use_features:
            n_features = u_features.shape[1] + v_features.shape[1]
        else:
            u_features, v_features = None, None
            n_features = 0
        all_indices = (u_indices, v_indices)

        all_indices_dual = np.concatenate((all_indices[0].reshape(-1, 1), all_indices[1].reshape(-1, 1)),
                                          axis=1)
        all_indices_dual = np.concatenate((all_indices_dual, labels.reshape(-1, 1)), axis=1)  # 合并labels

        n_dis = len(set(all_indices_dual[:, 0]))
        all_indices_dual[:, 1] += n_dis

        nodes = dict()
        nodes_l, nodes_d = np.unique(all_indices_dual[:, 0]), np.unique(all_indices_dual[:, 1])
        num_l, num_d = len(nodes_l), len(nodes_d)
        nodes['l'], nodes['d'], nodes['n_l'], nodes['n_d'] = nodes_l, nodes_d, num_l, num_d

        from bipartite import *

        Graph_lncRNA_disease = construct_D_lncRNA_disease(all_indices_dual,
                                                          nodes)  # ['base','0','1','2','3','4']
        Bi_graph = Graph_lncRNA_disease['base']
        print(Graph_lncRNA_disease.keys())
        Hs = generate_incidence_matrix_multiple(Graph_lncRNA_disease)  # 生成交互邻接矩阵
        del Graph_lncRNA_disease, Bi_graph

        # 开始处理正负样本问题
        samples = dict()
        pos_indices = np.where(all_indices_dual[:, -1] == 1)
        neg_indices = np.where(all_indices_dual[:, -1] == 0)
        samples['pos_samples'] = all_indices_dual[pos_indices]
        samples['neg_samples'] = all_indices_dual[neg_indices]

        args.conv = "sym"
        DG_l, DG_d = split_Gs(Hs, num_l)  # 将两类邻接矩阵分开，这里得到的两个邻接矩阵非方阵，因其没有指向同类node的边
        G_l = generate_Gs_from_O(args, DG_l)
        G_d = generate_Gs_from_O(args, DG_d)  # G_l/G_d:经归一化后的邻接矩阵，是方阵
        Gs_l, Gs_d = dict(), dict()  # 存储归一化后的邻接矩阵A
        Gcns_l, Gcns_d = dict(), dict()  # 存交互信息矩阵H
        for key, val in G_l.items():
            Gs_l[key] = torch.Tensor(G_l[key]).to(device)  # 归一化的邻接矩阵存储
            Gcns_l[key] = torch.Tensor(DG_l[key]).to(device)  # 原始邻接矩阵存储
        for key, val in G_d.items():
            Gs_d[key] = torch.Tensor(G_d[key]).to(device)
            Gcns_d[key] = torch.Tensor(DG_d[key]).to(device)

        print('begin constructing all_graphs')
        all_graphs = extracting_subgraphs(net, all_indices, labels, hop, u_features, v_features, hop * 2 + 1)
        random.shuffle(all_graphs)  # 重要！这里没有shuffle等于白给了
        mydataset = MyDataset(all_graphs, root='data/{}{}/{}/train'.format(*data_combo))

        print(f"测试数据将被分为{args.split_num}等份...")
        test_dataset = TestGraphDataset(u_indices, v_indices, all_indices, net, hop, u_features, v_features,
                                        args.split_num)

        print('constructing training graphs end.')

        sum = 0
        all_results = []
        max_f1 = 0
        for count in range(1):
            if args.dataset == 'Dataset1':
                feat_dim = 243
            else:
                feat_dim = 1178
            # K-fold cross-validation
            K = 5
            all_f1_mean, all_f1_std = 0, 0
            all_accuracy_mean, all_accuracy_std = 0, 0
            all_recall_mean, all_recall_std = 0, 0
            all_precision_mean, all_precision_std = 0, 0
            all_auc_mean, all_auc_std = 0, 0
            all_aupr_mean, all_aupr_std = 0, 0
            truth = []
            predict = []
            f1_s = []
            accuracy_s = []
            recall_s = []
            precision_s = []
            auc_s = []
            aupr_s = []
            max = 0
            all_truth = []
            all_predict = []

            fpr_list = []
            tpr_list = []
            precision_list = []
            recall_list = []
            for i in range(K):
                is_last_turn = i == K - 1
                print('*' * 25, i + 1, '*' * 25)
                model = MGSGCN(feat_dim, side_features=args.use_features, n_side_features=feat_dim,
                                node_dims=(num_l, num_d))
                train_graphs, test_graphs = get_k_fold_data(K, i, mydataset)
                test_auc, f1, accuracy, recall, precision, auc, aupr, one_truth, one_predict, one_pred_logits, trained_model, cos_final_matrix \
                    , fpr, tpr, precision_v, recall_v = \
                    train_multiple_epochs(train_graphs, test_graphs, model, Gcns_l, Gcns_d, test_dataset, is_last_turn)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                precision_list.append(precision_v)
                recall_list.append(recall_v)
                truth.extend(one_truth)
                predict.extend(one_predict)
                f1_s.append(f1)
                accuracy_s.append(accuracy)
                recall_s.append(recall)
                precision_s.append(precision)
                auc_s.append(auc)
                aupr_s.append(aupr)
                #all_truth.append(one_truth)
                #all_predict.append(one_predict)
                del model, train_graphs, test_graphs
                torch.cuda.empty_cache()
                gc.collect

            print('#' * 10, 'Final k-fold cross validation results', '#' * 10)
            print('The %d-fold CV auc: %f +/- %f' % (i, np.mean(auc_s), np.std(auc_s)))
            print('The %d-fold CV aupr: %f +/- %f' % (i, np.mean(aupr_s), np.std(aupr_s)))
            print('The %d-fold CV f1-score: %f +/- %f' % (i, np.mean(f1_s), np.std(f1_s)))
            print('The %d-fold CV recall: %f +/- %f' % (i, np.mean(recall_s), np.std(recall_s)))
            print('The %d-fold CV accuracy: %f +/- %f' % (i, np.mean(accuracy_s), np.std(accuracy_s)))
            print('The %d-fold CV precision: %f +/- %f' % (i, np.mean(precision_s), np.std(precision_s)))
            print('The %d-fold CV pred_logits: ', one_pred_logits.shape)

            all_f1_mean = all_f1_mean + np.mean(f1_s)
            all_f1_std = all_f1_std + np.std(f1_s)
            all_recall_mean = all_recall_mean + np.mean(recall_s)
            all_recall_std = all_recall_std + np.std(recall_s)

            all_accuracy_mean = all_accuracy_mean + np.mean(accuracy_s)
            all_accuracy_std = all_accuracy_std + np.std(accuracy_s)

            all_precision_mean = all_precision_mean + np.mean(precision_s)
            all_precision_std = all_precision_std + np.std(precision_s)

            all_auc_mean = all_auc_mean + np.mean(auc_s)
            all_auc_std = all_auc_std + np.std(auc_s)

            all_aupr_mean = all_aupr_mean + np.mean(aupr_s)
            all_aupr_std = all_aupr_std + np.std(aupr_s)

            #truth_predict = [truth, predict]
            #all_results.append(truth_predict)
            # 画图
            import matplotlib.pyplot as plt
            i = 0
            for fpr, tpr, score in zip(fpr_list, tpr_list, auc_s):
                plt.plot(fpr, tpr, label="Roc fold %d(AUC=%.4f)" % (i, score))
                i += 1
            plt.plot([0, 1], [0, 1], 'k--')  # 添加对角线
            plt.axis("square")
            plt.legend(fontsize=fontsize)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Dataset2 ROC and AUC")
            plt.savefig("roc2.tif", format="tif")
            plt.show()
            plt.clf()
            i = 0
            for precision, recall, score in zip(precision_list, recall_list, aupr_s):
                plt.plot(recall, precision, label="AUPR fold %d(AUPR=%.4f)" % (i,score))
                i += 1
            plt.legend(fontsize=fontsize)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Dataset2 Precision-Recall Curve")
            plt.savefig("aupr2.tif", format="tif")
            plt.show()
            i += 1

        truth_predict_array = np.array([truth, predict])
        tp_data_save_file = os.path.join("%s_graph" % dataset, args.feature_name)
        np.save(tp_data_save_file, truth_predict_array)
        print("save truth_predict_array to %s.npz" % tp_data_save_file)

        auc_aupr_array = np.array([np.mean(auc_s), np.mean(aupr_s)])
        score_data_save_file = os.path.join("%s_score" % dataset, args.feature_name)
        np.save(score_data_save_file, auc_aupr_array)
        print("save score_array to %s.npz" % score_data_save_file)


        torch.save(trained_model, '%s-%s-model.pth' % (dataset, feature_name))
        np.save(f'results/{args.dataset}_cos_final_matrix.npy', cos_final_matrix)

    print("All end...")