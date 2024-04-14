import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch



#loader = DataLoader(dataset=train_set, batch_size= params['batch_size'], shuffle=False, collate_fn=visit_collate_fn, num_workers=params['threads'])
#batch = next(iter(loader))
#inputs, targets, lengths, sorted_indice, ids = batch
#labels = feature_train.columns[2:]

def make_contribution_plot(z, model, inputs, labels, lengths, alpha, beta, num_features=34, num_seq=4):

    contr = np.array([[(alpha[z, j, 0] * torch.dot(model.output[1].weight[0],
                                                   torch.mul(beta[z, j],
                                                   model.embedding[0].weight[:, k])) * inputs[
                            z, j, k]).item() for k in range(num_features)] for j in range(num_seq)])

    fig, ax1 = plt.subplots(figsize=(15,12))
    plt.rcParams.update({'font.size':14,})
    plt.rcParams['axes.labelsize']=18
    plt.rcParams['lines.linewidth']=2
    plt.rcParams['font.size'] = 11
    colors = plt.cm.tab20(np.linspace(0, 1, num_features))
    cmap = ListedColormap(colors)

    ax2 = ax1.twinx()
    ax2.bar(np.arange(lengths[z]), alpha[z].detach().cpu().numpy().squeeze()[:lengths[z]][::-1], alpha=0.4,color='grey', width=0.6)
    ax2.set_ylabel('Visit Contribution', fontsize=18)


    for i in range(num_features):
        ax1.scatter(np.arange(lengths[z]), contr[:lengths[z], i][::-1], color=cmap(i))
        ax1.plot(np.arange(lengths[z]), contr[:lengths[z], i][::-1], label=labels[i], color=cmap(i))

    ax1.set_xlabel('time', fontsize=18)
    ax1.set_ylabel('Feature Contribution', fontsize=18)
    fig.tight_layout()
    plt.xlim(-0.5, 3.5)
    plt.xticks([0,1,2,3], labels=['base', 'week4', 'week8', 'week12'])
    ax1.legend(loc='center left', ncol=1, bbox_to_anchor=(1.08, 0.5))
    ax1.axhline(y=0, color='b', linestyle='-')
    plt.subplots_adjust(right=0.80)
    plt.savefig('results/contr_reg.jpeg', dpi=500)
    plt.show()




#labels = feature_train.columns[2:]
def make_global_contr_plots(model, inputs, labels, alpha, beta, num_features=68, num_seq=4):

    num_subjects = alpha.shape[0]
    sum_contr = np.zeros((num_seq, num_features))
    for z in range(num_subjects):
        contr = np.array([[(alpha[z, j, 0] * torch.dot(model.output[1].weight[0],
                                                       torch.mul(beta[z, j],
                                                                 model.embedding[0].weight[:, k])) * inputs[
                                z, j, k]).item() for k in range(num_features)] for j in range(num_seq)])
        sum_contr += np.abs(contr)
    avg_contr = sum_contr / num_subjects

    fig, ax1 = plt.subplots(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14, })
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.size'] = 10
    colors = plt.cm.tab20(np.linspace(0, 1, num_features))
    cmap = ListedColormap(colors)
    treatments = ['Nortriptyline', 'Duloxetine', 'Pregabalin', 'Mexiletine']
    for i in range(64, 68):
        ax1.scatter(np.arange(4), avg_contr[:4, i][::-1])
        ax1.plot(np.arange(4), avg_contr[:4, i][::-1], label=treatments[3 - (67 - i)])
    ax1.legend(loc='center left', ncol=2, bbox_to_anchor=(1.08, 0.5))
    ax1.set_xlabel('time')
    ax1.set_ylabel('Patients tendencey to have pain')
    fig.tight_layout()
    plt.xlim(-0.5, 3.5)
    plt.xticks([0, 1, 2, 3], labels=['base', 'week4', 'week8', 'week12'])
    ax1.legend()
    plt.subplots_adjust(right=0.95)
    plt.savefig('results/contr_reg_treatments.jpeg', dpi=500)
    plt.show()



    num_features = 60
    fig, ax1 = plt.subplots(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14, })
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['font.size'] = 10
    colors = plt.cm.tab20(np.linspace(0, 1, num_features))
    cmap = ListedColormap(colors)

    for i in range(num_features):
        ax1.scatter(np.arange(4), avg_contr[:4, i][::-1], color=cmap(i))
        ax1.plot(np.arange(4), avg_contr[:4, i][::-1], label=labels[i], color=cmap(i))

    ax1.legend(loc='center left', ncol=2, bbox_to_anchor=(1.08, 0.5))
    ax1.set_xlabel('time')
    ax1.set_ylabel('General feature importance')
    fig.tight_layout()
    plt.xlim(-0.2, 3.2)
    plt.xticks([0, 1, 2, 3], labels=['base', 'week4', 'week8', 'week12'])
    ax1.legend(ncol=2)
    plt.subplots_adjust(right=0.95)
    plt.savefig('results/contr_reg_avg.jpeg', dpi=500)
    plt.show()

