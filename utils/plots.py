import numpy as np
import seaborn as sns
from math import ceil
import matplotlib.pyplot as plt


def correlation(dfs) ->None:
    mask=np.triu(np.ones_like(dfs.corr()))
    sns.heatmap(dfs.corr(),mask=mask,cmap='Dark2',annot=True)
    plt.show()
def distribution(dfs,features:list,kde:bool=False) ->None :
    n_cols = 2
    n_rows = ceil(len(features)/n_cols)
    fig1, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
    ax = ax.flatten()
    for i, feature in enumerate(features):
        if kde:
            sns.kdeplot(dfs[feature],ax=ax[i])
        else:
            sns.histplot(dfs[feature],ax=ax[i],bins=50)

        # remove axes to show only one at the end
        plot_axes = [ax[i]]
        handles = []
        labels = []
        for plot_ax in plot_axes:
            handles += plot_ax.get_legend_handles_labels()[0]
            labels += plot_ax.get_legend_handles_labels()[1]
            plot_ax.legend().remove()
    for i in range(i+1, len(ax)):
        ax[i].axis('off')
    fig1.suptitle(f'Dataset Feature Distributions-[hist]', ha='center',  fontweight='bold', fontsize=25)   
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=2)
         

def relation():
    pass

if __name__=='__main__':
    pass