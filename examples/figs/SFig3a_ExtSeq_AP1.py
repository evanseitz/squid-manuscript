import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import pandas as pd
import logomaker

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])
    
alphabet = ['A','C','G','T']
alpha = 'dna'


if 1:
    rankFolder = 'rank0_seq13748'
    startA = 120
    stopA = startA + 7


fig_pad = 50
model_pad = 100

    
avgFolder = 'SQUID_AP1_intra_mut0/pad%s' % fig_pad
avgDir = os.path.join(parentDir, 'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR/%s/ADD_A/ADD_wildtype' % avgFolder)
avg_ISM = pd.read_csv(os.path.join(avgDir, 'avg_ISM_A.csv'), index_col=0)
avg_dL = pd.read_csv(os.path.join(avgDir, 'avg_other_A.csv'), index_col=0)
avg_add = pd.read_csv(os.path.join(avgDir, 'avg_additive_A.csv'), index_col=0)

wtFolder = 'SQUID_AP1_intra_mut0'
wtDir = os.path.join(parentDir, 'examples_DeepSTARR/c_surrogate_outputs/model_DeepSTARR/%s/%s' % (wtFolder, rankFolder))
#wt_ISM = pd.read_csv(os.path.join(wtDir, 'attributions_ISM_single.csv'), index_col=0)
wt_ISM = np.load(os.path.join(wtDir, 'attributions_ISM_single.npy'))
wt_ISM = squid_utils.arr2pd(wt_ISM, alphabet)
wt_dL = pd.read_csv(os.path.join(wtDir, 'attributions_deepLIFT_hypothetical.csv'), index_col=0)
wt_add = pd.read_csv(os.path.join(wtDir, 'mavenn_additive.csv'), index_col=0)

c_span = 'gray' #'magenta'

fig, axs = plt.subplots(3, 2, figsize=[15,4], gridspec_kw={'hspace': 0})#, 'wspace': 0})

ax = axs[0,0]
ax.set_title(r'Average of 50 $AP1$ attribution maps per method', fontsize=16)
logomaker.Logo(df=avg_ISM,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.set_ylabel('ISM', fontsize=14)

ax = axs[1,0]
logomaker.Logo(df=avg_dL,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.set_ylabel('DeepLIFT', fontsize=14)

ax = axs[2,0]
logomaker.Logo(df=avg_add,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.set_ylabel('Additive', fontsize=14)


ax = axs[0,1]
#ax.set_title(r'$Nanog$ attribution map from chr1:181653146\textendash 181654146', fontsize=16)
wt_ISM_norm = squid_utils.normalize(wt_ISM[startA-fig_pad:stopA+fig_pad], wt_ISM[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_ISM_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')

ax = axs[1,1]
wt_dL_norm = squid_utils.normalize(wt_dL[startA-fig_pad:stopA+fig_pad], wt_dL[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_dL_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')

ax = axs[2,1]
wt_add_norm = squid_utils.normalize(wt_add[startA-fig_pad:stopA+fig_pad], wt_add[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_add_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')


plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'compare_AP1_%s.png' % rankFolder), facecolor='w', dpi=600)
plt.show()
