import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from ranger import Ranger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def get_k_fold_data(k,i,data):
    assert k>1
    half_num, total_num = len(data) // 2, len(data)
    data_pos=data[0: half_num]
    data_neg=data[half_num:total_num]

    start=int(i*half_num//k)
    end=int((i+1)*half_num//k)

    data_valid_pos=data_pos[start:end]
    data_train_pos=data_pos[0:start]+data_pos[end:half_num]
    data_valid_neg=data_neg[start:end]
    data_train_neg=data_neg[0:start]+data_neg[end:half_num]
    data_train=data_train_pos+data_train_neg
    data_valid=data_valid_pos+data_valid_neg
    return data_train,data_valid

def make_ouput_logits(out, bar=0.5):
    tensor_min = out.min()
    tensor_max = out.max()
    tensor_norm = (out - tensor_min) / (bar * (tensor_max - tensor_min))
    return tensor_norm

def cos_sim_calculate(node_embeds, m, n):
    res = []
    sep_pos = node_embeds.shape[1] // 2
    for one_emb in node_embeds:
        # 将向量重塑为2D数组，因为cosine_similarity期望2D数组
        vector_a = one_emb[:sep_pos].reshape(1, -1)
        vector_b = one_emb[sep_pos:].reshape(1, -1)
        # 计算余弦相似度
        #cosine_sim = cosine_similarity(vector_a, vector_b)
        cosine_sim = (cosine_similarity(vector_a, vector_b) + 1) / 2
        res.append(cosine_sim.item())
    return np.array(res).reshape(m, n)


def train_multiple_epochs(train_graphs, test_graphs, model, H_l, H_d, real_test_dataset, is_last_turn):
    '''
    :param train_graphs:
    :param test_graphs:
    :param model:
    :param H_l: lncRna-dis交互信息矩阵
    :param H_d: dis-lncRna交互信息矩阵
    :return:
    '''
    print("starting train...")
    batch_size=64
    epochs=30
    train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size, shuffle=True, num_workers=0)
    optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=0)
    start_epoch = 1
    pbar = tqdm(range(start_epoch, epochs + start_epoch))

    for epoch in pbar:
        total_loss=0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            model.to(device)
            data = data.to(device)
            for key, value in H_l.items():
                H_l[key] = value.to(device)
            for key, value in H_d.items():
                H_d[key] = value.to(device)
            out = model(data, H_l, H_d)
            loss=F.cross_entropy(out, data.y.view(-1).long())
            loss.backward()
            total_loss+=loss.item()*num_graphs(data)
            optimizer.step()
        train_loss=total_loss/len(train_loader.dataset)
        train_auc=evaluate(model,train_loader,1, H_l, H_d)
        print('\n Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}'.format(epoch, train_loss, train_auc))

    ##################################
    cos_sim_matrix = None
    if is_last_turn:
        # 使用DataLoader来迭代数据集
        loader = DataLoader(real_test_dataset)  # batch_size设为None是因为IterableDataset自己控制batch
        pred_result = []
        node_embeddings_all = []
        for data in loader:
            # 这里处理你的数据
            print('分批处理结果中...')
            one_pred_result, node_embeddings = predict_test_batch(model, data, 2, H_l, H_d, True)
            pred_result.append(one_pred_result)
            node_embeddings_all.append(node_embeddings)

        pred_result_matrix = np.concatenate(pred_result, axis=1)
        node_embeddings_matrix = torch.cat(node_embeddings_all)
        truth, predict = pred_result_matrix[1], pred_result_matrix[0]

        vmax = max(predict)
        vmin = min(predict)

        alpha = 0.8
        predict_f1 = [0 for x in range(len(predict))]
        for p in range(len(predict)):
            predict_f1[p] = (predict[p] - vmin) / (vmax - vmin)
        predict_f1 = [int(item > alpha) for item in predict_f1]

        f1 = metrics.f1_score(truth, predict_f1)
        accuracy = metrics.accuracy_score(truth, predict_f1)
        recall = metrics.recall_score(truth, predict_f1)
        precision = metrics.precision_score(truth, predict_f1)
        fpr, tpr, thresholds1 = metrics.roc_curve(truth, predict, pos_label=1)
        print(len(predict), "predict:", predict[:10])
        auc_score = metrics.auc(fpr, tpr)
        p, r, thresholds2 = metrics.precision_recall_curve(truth, predict, pos_label=1)
        aupr_score = metrics.auc(r, p)
        print('f1:', f1)
        print('accuracy:', accuracy)
        print('recall:', recall)
        print('precision:', precision)
        print('auc:', auc_score)
        print('aupr:', aupr_score)
        cos_sim_matrix = cos_sim_calculate(node_embeddings_matrix, H_l['base'].shape[0], H_l['base'].shape[1])

    test_auc,one_pred_result=evaluate(model,test_loader,2, H_l, H_d)
    truth=one_pred_result[1]
    predict=one_pred_result[0]
    vmax=max(predict)
    vmin=min(predict)

    alpha=0.8
    predict_f1=[0 for x in range(len(predict))]
    for p in range(len(predict)):
        predict_f1[p]=(predict[p]-vmin)/(vmax-vmin)
    predict_f1=[int(item>alpha) for item in predict_f1]
    
    f1=metrics.f1_score(truth,predict_f1)
    accuracy=metrics.accuracy_score(truth,predict_f1)
    recall=metrics.recall_score(truth,predict_f1)
    precision=metrics.precision_score(truth,predict_f1)    
    fpr,tpr, thresholds1=metrics.roc_curve(truth,predict,pos_label=1)
    auc_score=metrics.auc(fpr,tpr)
    p,r,thresholds2=metrics.precision_recall_curve(truth,predict,pos_label=1)
    aupr_score=metrics.auc(r,p)
    print('f1:',f1)
    print('accuracy:',accuracy)
    print('recall:',recall)
    print('precision:',precision)
    print('auc:',auc_score)
    print('aupr:',aupr_score)
    #pred_logits = make_ouput_logits(predict, bar=alpha)
    pred_logits = (make_ouput_logits(predict, bar=alpha) + 1) / 2
    return test_auc, f1,accuracy,recall,precision,auc_score,aupr_score,truth,predict, pred_logits, model, cos_sim_matrix,fpr,tpr,p,r

def evaluate(model,loader,flag, H_l, H_d, need_embedding=False):
    one_pred_result=[]
    model.eval()
    predictions=torch.Tensor()
    labels=torch.Tensor()
    with torch.no_grad():
        for data in loader:
            model=model.to(device)
            data=data.to(device)
            pred = model(data, H_l, H_d, need_embedding=need_embedding)
            pred = torch.nn.functional.softmax(pred, dim=1)
            predictions = predictions.to(device)
            predictions=torch.cat((predictions,pred[:,1].detach()),0)
            predictions = predictions.to(device)
            labels = labels.to(device)
            labels=torch.cat((labels,data.y),0)

    if flag==2:
        one_pred_result=np.vstack((predictions,labels))

    fpr,tpr,_=metrics.roc_curve(labels,predictions,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    if flag==1:
        return auc
    else:
        return auc, one_pred_result


def predict_test(model, loader, flag, H_l, H_d, need_embedding=True):
    one_pred_result = []
    total_embeds = []
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if need_embedding:
                pred, embeds = model(data, H_l, H_d, need_embedding=need_embedding)
                total_embeds.append(embeds)
            else:
                pred = model(data, H_l, H_d, need_embedding=need_embedding)
            predictions = predictions.to(device)
            predictions = torch.cat((predictions, pred[:, 1].detach()), 0)
            labels = labels.to(device)
            labels = torch.cat((labels, data.y), 0)
        embeddings = torch.cat(total_embeds, 0)

    if flag == 2:
        one_pred_result = np.vstack((predictions, labels))

    fpr, tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if flag == 1:
        return auc
    else:
        return auc, one_pred_result, embeddings

def predict_test_batch(model, loader, flag, H_l, H_d, need_embedding=True):
    total_embeds = []
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if need_embedding:
                pred, embeds = model(data, H_l, H_d, need_embedding=need_embedding)
                total_embeds.append(embeds)
            else:
                pred = model(data, H_l, H_d, need_embedding=need_embedding)
            predictions=predictions.to(device)
            predictions = torch.cat((predictions, pred[:, 1].detach()), 0)
            labels = labels.to(device)
            labels = torch.cat((labels, data.y), 0)
        print("239:", need_embedding, len(total_embeds))
        embeddings = torch.cat(total_embeds, 0)
    one_pred_result = np.vstack((predictions, labels))
    return one_pred_result, embeddings

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def get_attention_matrix(model, data):
    edge_index = data.edge_index
    x = data.x

    _, edge_attention = model.message_and_save_attention(edge_index[0], x[edge_index[0]], x[edge_index[1]], None)

    num_nodes = data.num_nodes
    attention_matrix = torch.zeros(num_nodes, num_nodes)

    for i in range(edge_index.shape[1]):
        attention_matrix[edge_index[0, i], edge_index[1, i]] = edge_attention[i]
        attention_matrix[edge_index[1, i], edge_index[0, i]] = edge_attention[i]  # if the graph is undirected

    return attention_matrix





