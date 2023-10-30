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


trial = 24 #version of trained model
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


for df_idx in [20]:#range(0, seq_total):
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


fig = plt.figure(figsize=[15,7])#, constrained_layout=True)

gs1 = GridSpec(1, 3, left=0.05, right=0.97, bottom=0.5, top=0.97, wspace=0.2, hspace=0.1) #0.05, 0.95

pos = [1, 2, 3, 4]

# plot logo comparisons:
gs2 = GridSpec(4, 1, left=0.05, right=0.95, bottom=0.05, top=0.4, wspace=0.1, hspace=0) #0.05, 0.95

ax3 = fig.add_subplot(gs2[0, 0])
ax4 = fig.add_subplot(gs2[1, 0])
ax5 = fig.add_subplot(gs2[2, 0])
ax6 = fig.add_subplot(gs2[3, 0])

add_sort = np.argsort(dists_addFinal_addEarly) #86, 20, 24, 30, 75, 81, 73, 76, 89, 29, 57, 27, ...
plot_idx = 20

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