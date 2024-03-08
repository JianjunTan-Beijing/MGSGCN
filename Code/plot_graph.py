# -*- coding: utf-8 -*-
# 画图
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import argparse
feature_name_list = [
    "DSS+LFS",
    "DSS+DGS+LFS",
    "DSS+DGS+LFS+LGS",
    "DSS+DGS+DCS+LFS+LGS",
    "DSS+DGS+DCS+LFS+LGS+LCS"
]

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset name', default='Dataset1')  # Dataset1, Dataset2 数据集
args = parser.parse_args()
dataset = args.dataset
feature_name2score = {}

graph_dir = "%s_graph" % dataset
fontsize=10
feature_name2data = dict()
for feature_name in feature_name_list:
    data = np.load("%s/%s.npy" % (graph_dir, feature_name))
    feature_name2data[feature_name] = data
    score_data = np.load("%s/%s.npy" % ("%s_score" % dataset, feature_name))
    feature_name2score[feature_name] = score_data

for feature_name in feature_name_list:
    data = feature_name2data[feature_name]
    truth, predict = data[0], data[1]
    fpr, tpr, thresholds1 = metrics.roc_curve(truth, predict, pos_label=1)

    score = feature_name2score[feature_name][0]
    #plt.plot(fpr, tpr, label="%s (AUC=%.4f)" % (feature_name, score))
    label_text = "%s (AUC=%.4f)" % (feature_name, score)
    if feature_name == "DSS+LFS":
        label_text = "DSS+LFS (AUC=%.4f)" % score
    elif feature_name == "DSS+DGS+LFS":
        label_text = "DSGS+LFS (AUC=%.4f)" % score
    elif feature_name == "DSS+DGS+LFS+LGS":
        label_text = "DSGS+LFGS (AUC=%.4f)" % score
    elif feature_name == "DSS+DGS+DCS+LFS+LGS":
        label_text = "DSGCS+LFGS (AUC=%.4f)" % score
    elif feature_name == "DSS+DGS+DCS+LFS+LGS+LCS":
        label_text = "DSGCS+LFGCS (AUC=%.4f)" % score
    plt.plot(fpr, tpr, label=label_text, linewidth=1.8)
#plt.axis("square")
plt.legend(fontsize=fontsize)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("%s ROC and AUC" % dataset)
plt.savefig("%s/%s_roc.tif" % (graph_dir, dataset), dpi=300)
plt.show()
plt.clf()

for feature_name in feature_name_list:
    data = feature_name2data[feature_name]
    truth, predict = data[0], data[1]
    precision,recall,thresholds2=metrics.precision_recall_curve(truth,predict,pos_label=1)
    score = feature_name2score[feature_name][1]
    #plt.plot(recall, precision, label="%s (AUPR=%.4f)" % (feature_name,score))
    label_text = "%s (AUPR=%.4f)" % (feature_name, score)
    if feature_name == "DSS+LFS":
        label_text = "DSS+LFS (AUPR=%.4f)" % score
    elif feature_name == "DSS+DGS+LFS":
        label_text = "DSGS+LFS (AUPR=%.4f)" % score
    elif feature_name == "DSS+DGS+LFS+LGS":
        label_text = "DSGS+LFGS (AUPR=%.4f)" % score
    elif feature_name == "DSS+DGS+DCS+LFS+LGS":
        label_text = "DSGCS+LFGS (AUPR=%.4f)" % score
    elif feature_name == "DSS+DGS+DCS+LFS+LGS+LCS":
        label_text = "DSGCS+LFGCS (AUPR=%.4f)" % score
    plt.plot(recall, precision, label=label_text, linewidth=1.8)
plt.legend(fontsize=fontsize)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("%s Precision-Recall Curve" % dataset)
plt.savefig("%s/%s_aupr.tif" % (graph_dir, dataset), dpi=300)
plt.show()