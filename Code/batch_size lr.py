import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter

#auc_values = np.array([[0.9387, 0.9576, 0.9592, 0.9589],
                       #[0.9416, 0.9558, 0.9588, 0.9574],
                       #[0.9564, 0.9542, 0.9499, 0.9403]])
auc_values = np.array([[0.9837, 0.9685, 0.9783, 0.9338],
                       [0.9879, 0.9883, 0.9840, 0.90481],
                       [0.9634, 0.9120, 0.8454, 0.7737]])
# Corresponding learning rates and batch sizes
#learning_rates = [0.001, 0.0001, 0.00001]
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64, 128]

font = FontProperties(fname='path/to/times_new_roman.ttf', size=16)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(auc_values, annot=True, fmt=".4f", cmap="YlGnBu",
                 xticklabels=batch_sizes, yticklabels=learning_rates)

annot_font_size = 14
ax.set_yticklabels(ax.get_yticklabels(), fontsize=annot_font_size)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=annot_font_size)

y_labels = ['0.01', '0.001', '0.0001']
ax.set_yticklabels(y_labels)

plt.xlabel('batch_size', fontsize=14)
plt.ylabel('lr', fontsize=14)

plt.tick_params(labelsize=14)

plt.savefig('xuancan1.tif', format='tif', dpi=300)
plt.show()