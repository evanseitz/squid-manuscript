#env: use mavenn_citra

import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mavenn
import logomaker
import itertools
from scipy import stats


pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandparentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandparentDir)
import squid.utils as squid_utils

# User parameters:
N = 200000 #total number of sequences
motifA_name = 'Nanog'
motifB_name = 'Sox2'
start, stop = 480, 535
dist_range = np.arange(8,31,1) #inter-motif distances
alphabet = ['A','C','G','T']

dataDir = os.path.join(parentDir, 'global/outputs/%s%s/MAVENN' % (motifA_name, motifB_name))
logoDir = os.path.join(parentDir, 'global/outputs/%s%s/logos' % (motifA_name, motifB_name))
graphDir = os.path.join(parentDir, 'global/outputs/%s%s/graphs' % (motifA_name, motifB_name))
matrixDir = os.path.join(parentDir, 'global/outputs/%s%s/matrices' % (motifA_name, motifB_name))
if not os.path.exists(logoDir):
    os.mkdir(logoDir)
if not os.path.exists(graphDir):
    os.mkdir(graphDir)
if not os.path.exists(matrixDir):
    os.mkdir(matrixDir)


if 1: #analyze additive matrices
    additive_sums = []
    additive_matrices = np.zeros(shape=(stop-start, 4, len(dist_range)))
    additive_graph = True
    for idx, inter_dist in enumerate(dist_range):
        fname = '%s%s_N%s_dist%s_GE_pairwise' % (motifA_name, motifB_name, N, inter_dist)
        model = mavenn.load(os.path.join(dataDir, fname))
        theta_dict = model.get_theta(gauge='empirical')
        theta_dict.keys()
        theta_lc = theta_dict['theta_lc']*-1.
        #theta_lc = squid_utils.fix_gauge(theta_lc, 'hierarchical', wt=None)
        additive_matrices[:,:,idx] = theta_lc
        additive_sums.append(np.sum(theta_lc, axis=(0,1)))

        if 0:
            print('Saving distance %s logo...' % inter_dist)
            fig, ax = plt.subplots(figsize=[10.5,2]) #[15,2]
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            logomaker.Logo(df=squid_utils.arr2pd(additive_matrices[:,:,idx], alphabet),
                            ax=ax,
                            fade_below=.5,
                            shade_below=.5,
                            width=.9,
                            center_values=True,
                            font_name='Arial Rounded MT Bold')
            ax.set_ylim(-8.5, 8.5)
            plt.tight_layout()
            plt.savefig(os.path.join(logoDir,'logo_dist%s.png' % inter_dist), facecolor='w', dpi=600)
            #plt.show()
            plt.close()


if 1: #analyze pairwise matrices
    PW_absval = []
    PW_sums = []
    PW_sums_p = []
    PW_sums_n = []
    PW_sums_mask = []
    # loop to find total max and min across all pairwise matrices:
    for idx, inter_dist in enumerate(dist_range):
        PW_sum_boxes = []
        PW_sum_boxes_p = []
        PW_sum_boxes_n = []
        fname = '%s%s_N%s_dist%s_GE_pairwise' % (motifA_name, motifB_name, N, inter_dist)
        model = mavenn.load(os.path.join(dataDir, fname))
        theta_dict = model.get_theta(gauge='empirical') #fixes gauge
        theta_dict.keys()
        theta_lclc = theta_dict['theta_lclc']*-1.

        if 1: #integrate over interaction energy only v
            pairwise_graph = True
            MA = np.arange(14,15,1) #(9,19,1)
            MB = np.arange(11+inter_dist,21+inter_dist)
            MAxMB = [MA,MB]
            
            sum_MAxMB = 0
            for i,j in itertools.product(*MAxMB):
                if 0: #for visualizing mask only
                    theta_lclc[i,:,j,:] = 10
                sum_MAxMB += sum(map(sum,theta_lclc[i,:,j,:]))
            PW_sums_mask.append(sum_MAxMB)

        for i in range(np.shape(theta_lclc)[0]):
            for j in range(np.shape(theta_lclc)[0]):
                if i == (j-1):

                    temp1 = np.copy(theta_lclc[i,:,j,:])
                    temp1[temp1<0] = 0
                    temp2 = np.copy(theta_lclc[i,:,j,:])
                    temp2[temp2>0] = 0

                    PW_absval.append(np.amax(np.absolute(theta_lclc[i,:,j,:])))
                    PW_sum_boxes.append(np.sum(theta_lclc[i,:,j,:].flatten()))
                    PW_sum_boxes_p.append(np.sum(temp1.flatten()))
                    PW_sum_boxes_n.append(np.sum(temp2.flatten()))

        PW_sums.append(np.sum(PW_sum_boxes))
        PW_sums_p.append(np.sum(PW_sum_boxes_p))
        PW_sums_n.append(np.sum(PW_sum_boxes_n))
    PW_max = np.amax(PW_absval)

    if 0: #plot pairwise matrices
        for idx, inter_dist in enumerate(dist_range):
            fname = '%s%s_N%s_dist%s_GE_pairwise' % (motifA_name, motifB_name, N, inter_dist)
            model = mavenn.load(os.path.join(dataDir, fname))
            theta_dict = model.get_theta(gauge='empirical') #fixes gauge
            theta_dict.keys()
            theta_lclc = theta_dict['theta_lclc']*-1.

            print('Saving distance %s pairwise...' % inter_dist)
            fig, ax = plt.subplots(figsize=[10,5])
            ax, cb = mavenn.heatmap_pairwise(values=theta_lclc,
                                            alphabet=alphabet,
                                            ax=ax,
                                            gpmap_type='pairwise',
                                            cmap_size='2%',
                                            show_alphabet=False,
                                            cmap='seismic',
                                            cmap_pad=.1,
                                            show_seplines=True,            
                                            sepline_kwargs = {'color': 'k',
                                                                'linestyle': '-',
                                                                'linewidth': .5,
                                                                'color':'gray'})
            cb.set_label(r'Pairwise Effect', labelpad=8, ha='center', va='center', rotation=-90)
            cb.outline.set_visible(False)
            cb.ax.tick_params(direction='in', size=20, color='white')
            plt.cm.ScalarMappable.set_clim(cb, vmin=-1.*PW_max, vmax=PW_max)

            plt.tight_layout()
            plt.savefig(os.path.join(matrixDir,'pairwise_dist%s.png' % inter_dist), facecolor='w', dpi=600)
            #plt.show()
            plt.close()

if 1: #plot summation of model parameters vs inter_dist
    fig, ax = plt.subplots(2, figsize=(5,5)) #10.5, 4

    if additive_graph is True:
        z = np.polyfit(dist_range, additive_sums, 6)
        f = np.poly1d(z)
        x_new = np.linspace(dist_range[0], dist_range[-1], 200)
        y_new = f(x_new)
        ax[0].plot(x_new, y_new, c='gray')
        ax[0].scatter(dist_range, additive_sums, s=10, zorder=100, c='k')
        #ax[0].axvline(10, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        #ax[0].axvline(20, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        #ax[0].axvline(30, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        ax[0].set_xticks([], [])
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='both', which='major', labelsize=8)
        ax[0].tick_params(axis='both', which='minor', labelsize=8)

    if pairwise_graph is True:
        z = np.polyfit(dist_range, PW_sums_mask, 6)
        f = np.poly1d(z)
        x_new = np.linspace(dist_range[0], dist_range[-1], 200)
        y_new = f(x_new)
        ax[1].plot(x_new, y_new, c='gray')
        ax[1].scatter(dist_range, PW_sums_mask, s=8, zorder=100, c='k')
        #ax[1].axvline(10, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        #ax[1].axvline(20, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        #ax[1].axvline(30, c='lightgray', linestyle='--', linewidth=1, zorder=-10)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='both', which='major', labelsize=8)
        ax[1].tick_params(axis='both', which='minor', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(graphDir,'parameters_sum.png'), facecolor='w', dpi=600)
    #plt.show()
    plt.close()


    if additive_graph is True and pairwise_graph is True:
        fig, ax = plt.subplots()
        ax.scatter(additive_sums, PW_sums_mask)
        ax.set_title('Pearson: %s' % stats.pearsonr(additive_sums, PW_sums_mask)[0])
        plt.savefig(os.path.join(graphDir,'add_pw_corr.png'), facecolor='w', dpi=600)
        #plt.show()
        plt.close()


if 0: #plot close up of single pairwise matrix
    inter_dist = 16
    N = 500000
    fname = '%s%s_N%s_dist%s_GE_pairwise' % (motifA_name, motifB_name, N, 16)
    model = mavenn.load(os.path.join(dataDir, fname))
    theta_dict = model.get_theta(gauge='empirical') #fixes gauge
    theta_dict.keys()
    theta_lc = theta_dict['theta_lc']*-1.
    theta_lclc = theta_dict['theta_lclc']*-1.

    if 1:
        if 0:
            plt.hist(theta_lclc.flatten(), bins=100)
            plt.show()
        temp = theta_lclc.flatten()
        temp[(temp >= -.25) & (temp <= .25)] = 0 #remove noise based on threshold for visualization
        theta_lclc = temp.reshape(theta_lclc.shape)

    fig, ax = plt.subplots(figsize=[10.5,2]) #[15,2]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    logomaker.Logo(df=squid_utils.arr2pd(theta_lc[0:40,:], alphabet),
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    ax.set_ylim(-8.5, 8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(logoDir,'logo_dist%s_CU_500k.png' % inter_dist), facecolor='w', dpi=600)
    #plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=[10,5])
    ax, cb = mavenn.heatmap_pairwise(values=theta_lclc[0:40,:,0:40,:],
                                    alphabet=alphabet,
                                    ax=ax,
                                    gpmap_type='pairwise',
                                    cmap_size='2%',
                                    show_alphabet=False,
                                    cmap='seismic',
                                    cmap_pad=.1,
                                    show_seplines=True,            
                                    sepline_kwargs = {'color': 'k',
                                                        'linestyle': '-',
                                                        'linewidth': .5,
                                                        'color':'gray'})
    cb.set_label(r'Pairwise Effect', labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(matrixDir,'pairwise_dist%s_CU_500k.png' % inter_dist), facecolor='w', dpi=600)
    #plt.show()
    plt.close()