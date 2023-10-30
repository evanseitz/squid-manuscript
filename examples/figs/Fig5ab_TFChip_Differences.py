import os, sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker
from numpy import linalg as LA
from six.moves import cPickle
from scipy import stats
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

np.random.seed(0)

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])
    

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils


trial = 23 #version of trained model
seq_total = 100 #total number of sequences
alphabet = ['A','C','G','T']

earlyDir = os.path.join(parentDir, 'examples_TFChIP/c_surrogate_outputs/model_TFChIP_trial%s_early/SQUID_GABPA_intra_mut0' % trial)
finalDir = os.path.join(parentDir, 'examples_TFChIP/c_surrogate_outputs/model_TFChIP_trial%s_final/SQUID_GABPA_intra_mut0' % trial)

with open(os.path.join(parentDir,'examples_TFChIP/a_model_assets/gabpa_results.pickle'), 'rb') as fin:
    X = cPickle.load(fin)
    results = cPickle.load(fin)
    attr_ensemble = cPickle.load(fin)


# Distances from ___ to ___ :
dists_addFinal_addEarly = []
dists_salFinal_salEarly = []
dists_shapFinal_shapEarly = []
dists_sgFinal_sgEarly = []


for df_idx in range(0, seq_total):
    print('Sequence index:',df_idx)
    for folder in os.listdir(earlyDir):
        if folder.startswith('rank%s_' % (df_idx)):
            path = os.path.join(earlyDir, folder)
            add_early = pd.read_csv(os.path.join(path,'mavenn_additive.csv'), index_col=0)
    for folder in os.listdir(finalDir):
        if folder.startswith('rank%s_' % (df_idx)):
            path = os.path.join(finalDir, folder)
            add_final = pd.read_csv(os.path.join(path,'mavenn_additive.csv'), index_col=0)
            
    sal_early = squid_utils.arr2pd(results[trial]['saliency_early'][df_idx], alphabet)
    sal_final = squid_utils.arr2pd(results[trial]['saliency_final'][df_idx], alphabet)
    
    shap_early = squid_utils.arr2pd(results[trial]['shap_early'][df_idx], alphabet)
    shap_final = squid_utils.arr2pd(results[trial]['shap_final'][df_idx], alphabet)
    
    sg_early = squid_utils.arr2pd(results[trial]['sg_early'][df_idx], alphabet)
    sg_final = squid_utils.arr2pd(results[trial]['sg_final'][df_idx], alphabet)
    
    if 1: #normalize
        add_early = squid_utils.normalize(np.array(add_early), np.array(add_early))
        add_final = squid_utils.normalize(np.array(add_final), np.array(add_final))
        sal_early = squid_utils.normalize(np.array(sal_early), np.array(sal_early))
        sal_final = squid_utils.normalize(np.array(sal_final), np.array(sal_final))
        shap_early = squid_utils.normalize(np.array(shap_early), np.array(shap_early))
        shap_final = squid_utils.normalize(np.array(shap_final), np.array(shap_final))
        sg_early = squid_utils.normalize(np.array(sg_early), np.array(sg_early))
        sg_final = squid_utils.normalize(np.array(sg_final), np.array(sg_final))
    else:
        add_early = np.array(add_early)
        add_final = np.array(add_final)
        sal_early = np.array(sal_early)
        sal_final = np.array(sal_final)
        shap_early = np.array(shap_early)
        shap_final = np.array(shap_final)
        sg_early = np.array(sg_early)
        sg_final = np.array(sg_final)

    if 1: #gauge fix
        gauge = 'wildtype'
        add_early = squid_utils.fix_gauge(np.array(add_early), gauge=gauge, wt=X[df_idx])
        add_final = squid_utils.fix_gauge(np.array(add_final), gauge=gauge, wt=X[df_idx])
        sal_early = squid_utils.fix_gauge(np.array(sal_early), gauge=gauge, wt=X[df_idx])
        sal_final = squid_utils.fix_gauge(np.array(sal_final), gauge=gauge, wt=X[df_idx])
        shap_early = squid_utils.fix_gauge(np.array(shap_early), gauge=gauge, wt=X[df_idx])
        shap_final = squid_utils.fix_gauge(np.array(shap_final), gauge=gauge, wt=X[df_idx])
        sg_early = squid_utils.fix_gauge(np.array(sg_early), gauge=gauge, wt=X[df_idx])
        sg_final = squid_utils.fix_gauge(np.array(sg_final), gauge=gauge, wt=X[df_idx])


    dists_addFinal_addEarly.append(np.linalg.norm(add_final - add_early, axis=(0,1)))
    dists_salFinal_salEarly.append(np.linalg.norm(sal_final - sal_early, axis=(0,1)))
    dists_shapFinal_shapEarly.append(np.linalg.norm(shap_final - shap_early, axis=(0,1)))
    dists_sgFinal_sgEarly.append(np.linalg.norm(sg_final - sg_early, axis=(0,1)))


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
            
singles_y = [dists_addFinal_addEarly,
             dists_shapFinal_shapEarly,
             dists_sgFinal_sgEarly,
             dists_salFinal_salEarly]


if 1:
    mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[1], alternative='less')
    print('add_vs_shap p-value:', pval)
    mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[2], alternative='less')
    print('add_vs_sg p-value:', pval)
    mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[3], alternative='less')
    print('add_vs_sal p-value:', pval)


fig = plt.figure(figsize=[15,7])#, constrained_layout=True)

gs1 = GridSpec(1, 2, left=0.05, right=0.97, bottom=0.5, top=0.97, wspace=0.2, hspace=0.1) #0.05, 0.95

if 1:
    # plot Epoch vs AUROC:
    ax1 = fig.add_subplot(gs1[0, 0])
    #for t in range(25):
    train_auroc = results[trial]['train_auroc']
    val_auroc = results[trial]['val_auroc']
    early_stop_index = results[trial]['early_stop_index']
    final_stop_index = len(train_auroc)
    print(early_stop_index, final_stop_index)
    ax1.plot(train_auroc, 'r', label='train set')
    ax1.plot(val_auroc, 'b', label='valid set')
    ax1.set_xlabel('Epoch', fontsize=16, labelpad=-1)
    ax1.set_ylabel('AUROC', fontsize=16, labelpad=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.plot([early_stop_index, early_stop_index], [0.51, 1.], '--k')
    ax1.plot([final_stop_index, final_stop_index], [0.51, 1.], '--k')
    ax1.set_ylim([0.51, 1.0])
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_xticklabels(['','0','','','','200'])
    leg1 = ax1.legend(fontsize=14, frameon=False, loc='lower center')
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)


if 1: #plot box plots
    ax2 = fig.add_subplot(gs1[0, 1])
    flierprops = dict(marker=(3,0,0), markeredgecolor='k', markerfacecolor='k', markersize=10, linestyle='none')
    add_box = ax2.boxplot(singles_y[0], showfliers=False, vert=True, showmeans=True, meanprops=flierprops, positions=[1], widths=0.5)
    shap_box = ax2.boxplot(singles_y[1], showfliers=False, vert=True, showmeans=True, meanprops=flierprops, positions=[2], widths=0.5)
    sg_box = ax2.boxplot(singles_y[2], showfliers=False, vert=True, showmeans=True, meanprops=flierprops, positions=[3], widths=0.5)
    sal_box = ax2.boxplot(singles_y[3], showfliers=False, vert=True, showmeans=True, meanprops=flierprops, positions=[4], widths=0.5)

    colors = ['#4daf4a','#e41a1c','#a65628','#ff7f00'] #yellow : #ffff33 | pink : #f781bf
    set_box_color(add_box, colors[0])
    set_box_color(shap_box, colors[1]) 
    set_box_color(sg_box, colors[2])
    set_box_color(sal_box, colors[3])

    pos = [1, 2, 3, 4]
    labels = ['Additive', 'DeepSHAP', 'SmoothGrad', 'Saliency']
    s = 10 #scatter scale
    a = .2 #scatter alpha
    for i in range(0,len(singles_y)):
        singles_x = np.random.normal(pos[i], 0.08, size=seq_total) #0.05
        ax2.scatter(singles_x, singles_y[i], alpha=a, s=s, c=colors[i], label=labels[i], zorder=1)

    #ax2.set_xlabel('Attribution methods', fontsize=16, labelpad=16)
    ax2.set_ylabel('Difference', fontsize=16, labelpad=12) #'Difference in atribution maps'
    #ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.xaxis.set_ticklabels([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, ax2.get_ylim()[1]+1)
    ax2.set_xticklabels(['Additive', 'DeepSHAP', 'SmoothGrad', 'Saliency'], fontsize=16)
    ax2.tick_params(axis='x', which='major', pad=12)


    # add MWU information to plot
    alt = 'less' #{'two-sided', 'less'}
    A, B, C, D = pos[0], pos[1], pos[2], pos[3]
    combinations = [(A, B), (A, C), (A, D)]
    ax_bottom, ax_top = ax2.get_ylim()
    y_range = ax_top - ax_bottom
    ax_tops = []
    ax_bots = []
    bar_heights = []
    ax_tops.append([item.get_ydata()[1] for item in add_box['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in shap_box['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in sg_box['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in sal_box['caps']][1])
    ax_bots.append([item.get_ydata()[1] for item in add_box['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in shap_box['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in sg_box['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in sal_box['caps']][0])

    if 0:
        level_idx = 3
        for x1, x2 in combinations:
            if x1 == A and x2 == B:
                mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[1], alternative=alt)
            if x1 == A and x2 == C:
                mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[2], alternative=alt)
            elif x1 == A and x2 == D:
                mwu_stat, pval = stats.mannwhitneyu(singles_y[0], singles_y[3], alternative=alt)

            if pval < 0.001:
                sig_symbol = '***'
            elif pval < 0.01:
                sig_symbol = '**'
            elif pval < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = r'$\it{ns}$'
                
            bar_height = max(ax_tops) + (y_range * 0.07 * level_idx)
            bar_heights.append(bar_height)
            bar_tips = bar_height - (y_range * 0.02)
            ax2.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            text_height = bar_height + (y_range * 0.001)
            ax2.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
            level_idx -= 1

    for c in list(range(1, 5)):
        ax2.text(c, 0 + y_range * 0.02, r'$\it{n} =$ %s' % seq_total, ha='center', size='small')
    

# plot sequenc-function space schematic
'''if 0:
    ax3 = fig.add_subplot(gs1[0, 2])

    x_vals = np.arange(3.25, 4.5, 0.02)
    sparse = np.random.choice(np.shape(x_vals)[0],int(np.shape(x_vals)[0]/2.), replace=False)
    sparse = np.sort(sparse)
    y_vals = []
    y_vals_noise = []
    noise = np.random.normal(loc=0.0, scale=1.5, size=(x_vals[sparse].shape[0],))

    a = .2
    for i, x in enumerate(x_vals[sparse]):
        y = ((x-a)*(np.sin(np.pi*(x-a))+np.cos(2*np.pi*(x-a))))
        y_vals_noise.append(y + noise[i])

    for i, x in enumerate(x_vals):
        y = ((x-a)*(np.sin(np.pi*(x-a))+np.cos(2*np.pi*(x-a))))
        y_vals.append(y)

    ax3.plot(x_vals[sparse], y_vals_noise, c='k')
    ax3.scatter(x_vals[sparse], y_vals_noise, c='k', s=10)
    ax3.plot(x_vals, y_vals, c='gray', linestyle='--', zorder=-10)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])'''


if 0: #plot logo comparisons (see 'Fig5c_EarlyFinal_Logos.py' for plot used in paper)
    gs2 = GridSpec(4, 1, left=0.05, right=0.95, bottom=0.05, top=0.4, wspace=0.1, hspace=0) #0.05, 0.95

    ax3 = fig.add_subplot(gs2[0, 0])
    ax4 = fig.add_subplot(gs2[1, 0])
    ax5 = fig.add_subplot(gs2[2, 0])
    ax6 = fig.add_subplot(gs2[3, 0])

    add_sort = np.argsort(dists_addFinal_addEarly) #86, 20, 24, 30, 75, 81, 73, 76, 89, 29, 57, 27, ...
    plot_idx = 20
    if 0:
        ax2.scatter(pos[0],dists_addFinal_addEarly[plot_idx], c='k')
        ax2.scatter(pos[3],dists_salFinal_salEarly[plot_idx], c='k')

    for df_idx in [plot_idx]:
        for folder in os.listdir(earlyDir):
            if folder.startswith('rank%s_' % (df_idx)):
                path = os.path.join(earlyDir, folder)
                add_early = pd.read_csv(os.path.join(path,'mavenn_additive.csv'), index_col=0)
        for folder in os.listdir(finalDir):
            if folder.startswith('rank%s_' % (df_idx)):
                path = os.path.join(finalDir, folder)
                add_final = pd.read_csv(os.path.join(path,'mavenn_additive.csv'), index_col=0)

        sal_early = squid_utils.arr2pd(results[trial]['saliency_early'][df_idx], alphabet)
        sal_final = squid_utils.arr2pd(results[trial]['saliency_final'][df_idx], alphabet)
        
        if 1: #normalize
            add_early = squid_utils.normalize(np.array(add_early), np.array(add_early))
            add_final = squid_utils.normalize(np.array(add_final), np.array(add_final))
            sal_early = squid_utils.normalize(np.array(sal_early), np.array(sal_early))
            sal_final = squid_utils.normalize(np.array(sal_final), np.array(sal_final))
        else:
            add_early = np.array(add_early)
            add_final = np.array(add_final)
            sal_early = np.array(sal_early)
            sal_final = np.array(sal_final)

    center = True
    logomaker.Logo(df=squid_utils.arr2pd(sal_early, alphabet),
                    ax=ax3,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=center,
                    font_name='Arial Rounded MT Bold')
    ax3.axes.get_yaxis().set_ticks([])
    ax3.set_ylabel('Early', fontsize=12, labelpad=2)

    logomaker.Logo(df=squid_utils.arr2pd(sal_final, alphabet),
                    ax=ax4,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=center,
                    font_name='Arial Rounded MT Bold')
    ax4.axes.get_yaxis().set_ticks([])
    ax4.set_ylabel('Final', fontsize=12, labelpad=2)

    logomaker.Logo(df=squid_utils.arr2pd(add_early, alphabet),
                    ax=ax5,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=center,
                    font_name='Arial Rounded MT Bold')
    ax5.axes.get_yaxis().set_ticks([])
    ax5.set_ylabel('Early', fontsize=12, labelpad=2)

    logomaker.Logo(df=squid_utils.arr2pd(add_final, alphabet),
                    ax=ax6,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=center,
                    font_name='Arial Rounded MT Bold')
    ax6.axes.get_yaxis().set_ticks([])
    ax6.set_ylabel('Final', fontsize=12, labelpad=2)


plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'TFChIP_diffs.pdf'), facecolor='w', dpi=200)
plt.show()