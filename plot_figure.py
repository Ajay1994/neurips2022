import numpy as np
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as mtick
from mpl_toolkits import mplot3d
import seaborn as sns
# sns.set_style("whitegrid")
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(16,5))

ax = fig.add_subplot(1, 2, 1)
d = {'Method': ['Baseline', 'Dense-Sup', 'Baseline', 'Dense-Sup', 'Baseline', 'Dense-Sup'], 
     'Prune Ratio': ['85%', '85%', '90%', '90%', '95%', '95%'], 
     'Data':[91.13, 91.54, 90.59, 91.25, 89.95, 90.20]}
df = pd.DataFrame(data=d)
graph = sns.barplot(data=df, x='Prune Ratio', y='Data', hue='Method',  ci=None, palette=["crimson", "rosybrown" ,"steelblue"])
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(fontsize= 16)
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Sparsity Ratio (LTH)\nVgg-16 + CIFAR-10", fontsize=16)
plt.ylim([89,92.5])

ax = fig.add_subplot(1, 2, 2)
d = {'Method': ['Baseline', 'Dense-Sup', 'Baseline', 'Dense-Sup', 'Baseline', 'Dense-Sup'], 
     'Prune Ratio': ['85%', '85%', '90%', '90%', '95%', '95%'], 
     'Data':[90.26, 90.6, 89.65, 89.9, 90.25, 91.32]}
df = pd.DataFrame(data=d)
graph = sns.barplot(data=df, x='Prune Ratio', y='Data', hue='Method',  ci=None, palette=["crimson", "rosybrown" ,"steelblue"])
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.legend(fontsize= 16)
plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Sparsity Ratio (LTH)\nResNet50 + CIFAR-10", fontsize=16)
plt.ylim([89, 92])

plt.tight_layout()
plt.savefig("./plots/comparison.pdf")