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
gauge = 'wildtype' #trivial here
fig_pad = 50#45
model_pad = 100


if 0:
    startA = 507
    stopA = startA + 7
    rankFolder = 'rank6_seq3769'
elif 0:
    startA = 480
    stopA = startA + 7
    rankFolder = 'rank7_seq3569'
elif 0:
    startA = 498
    stopA = startA + 7
    rankFolder = 'rank10_seq8110'
elif 0:
    startA = 471
    stopA = startA + 7
    rankFolder = 'rank15_seq12856'
elif 0:
    startA = 498
    stopA = startA + 7
    rankFolder = 'rank19_seq11327'
elif 1:
    startA = 499
    stopA = startA + 7
    rankFolder = 'rank27_seq13216'
elif 0:
    startA = 535
    stopA = startA + 7
    rankFolder = 'rank30_seq22884'
elif 0:
    startA = 485
    stopA = startA + 7
    rankFolder = 'rank33_seq16430'
elif 0:
    startA = 472
    stopA = startA + 7
    rankFolder = 'rank36_seq15902'
elif 0:
    startA = 488
    stopA = startA + 7
    rankFolder = 'rank37_seq7150'
elif 0:
    startA = 488
    stopA = startA + 7
    rankFolder = 'rank42_seq24323'
elif 0:
    startA = 501
    stopA = startA + 7
    rankFolder = 'rank43_seq13785'
elif 0:
    startA = 537
    stopA = startA + 7
    rankFolder = 'rank49_seq25360'

    
avgFolder = 'SQUID_Nanog_intra_mut0/pad%s' % fig_pad
avgDir = os.path.join(parentDir, 'examples_BPNet/d_outputs_analysis/model_BPNet_OSKN/%s/ADD_A/ADD_%s' % (avgFolder, gauge))
avg_ISM = pd.read_csv(os.path.join(avgDir, 'avg_ISM_A.csv'), index_col=0)
avg_dL = pd.read_csv(os.path.join(avgDir, 'avg_other_A.csv'), index_col=0)
avg_add = pd.read_csv(os.path.join(avgDir, 'avg_additive_A.csv'), index_col=0)

wtFolder = 'SQUID_Nanog_intra_mut0'
wtDir = os.path.join(parentDir, 'examples_BPNet/c_surrogate_outputs/model_BPNet_OSKN/%s/%s' % (wtFolder, rankFolder))
#wt_ISM = pd.read_csv(os.path.join(wtDir, 'attributions_ISM_single.csv'), index_col=0)
wt_ISM = np.load(os.path.join(wtDir, 'attributions_ISM_single.npy'))
wt_ISM = squid_utils.arr2pd(wt_ISM, alphabet)
wt_dL = pd.read_csv(os.path.join(wtDir, 'attributions_deepLIFT_hypothetical.csv'), index_col=0)
wt_add = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)

c_span = 'gray' #'magenta'

fig, axs = plt.subplots(3, 2, figsize=[15,4], gridspec_kw={'hspace': 0})#, 'wspace': 0})

ax = axs[0,0]
ax.set_title(r'Average of 50 $Nanog$ attribution maps per method', fontsize=16)
logomaker.Logo(df=avg_ISM,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(8-.5+6, 9+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(18-.5+6, 19+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(28-.5+6, 29+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(38-.5+6, 39+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(60-.5+6, 61+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(70-.5+6, 71+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(80-.5+6, 81+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(90-.5+6, 91+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.set_ylabel('ISM', fontsize=14)

ax = axs[1,0]
logomaker.Logo(df=avg_dL,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(8-.5+6, 9+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(18-.5+6, 19+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(28-.5+6, 29+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(38-.5+6, 39+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(60-.5+6, 61+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(70-.5+6, 71+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(80-.5+6, 81+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(90-.5+6, 91+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.set_ylabel('DeepLIFT', fontsize=14)

ax = axs[2,0]
logomaker.Logo(df=avg_add,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(8-.5+6, 9+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(18-.5+6, 19+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(28-.5+6, 29+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(38-.5+6, 39+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(60-.5+6, 61+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(70-.5+6, 71+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(80-.5+6, 81+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(90-.5+6, 91+.5+6, alpha=.1, color=c_span, zorder=-10)
ax.set_ylabel('Additive', fontsize=14)


ax = axs[0,1]
ax.set_title(r'$Nanog$ attribution map from chr1:181653146\textendash 181654146', fontsize=16)
wt_ISM_norm = squid_utils.normalize(wt_ISM[startA-fig_pad:stopA+fig_pad], wt_ISM[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_ISM_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(startA-7-.5, startA-6+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-17-.5, startA-16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-27-.5, startA-26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-37-.5, startA-36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+15-.5, startA+16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+25-.5, startA+26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+35-.5, startA+36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+45-.5, startA+46+.5, alpha=.1, color=c_span, zorder=-10)

ax = axs[1,1]
wt_dL_norm = squid_utils.normalize(wt_dL[startA-fig_pad:stopA+fig_pad], wt_dL[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_dL_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(startA-7-.5, startA-6+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-17-.5, startA-16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-27-.5, startA-26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-37-.5, startA-36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+15-.5, startA+16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+25-.5, startA+26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+35-.5, startA+36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+45-.5, startA+46+.5, alpha=.1, color=c_span, zorder=-10)

ax = axs[2,1]
wt_add_norm = squid_utils.normalize(wt_add[startA-fig_pad:stopA+fig_pad], wt_add[startA-model_pad:stopA+model_pad])
logomaker.Logo(df=wt_add_norm,
                ax=ax,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')
ax.axvspan(startA-7-.5, startA-6+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-17-.5, startA-16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-27-.5, startA-26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA-37-.5, startA-36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+15-.5, startA+16+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+25-.5, startA+26+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+35-.5, startA+36+.5, alpha=.1, color=c_span, zorder=-10)
ax.axvspan(startA+45-.5, startA+46+.5, alpha=.1, color=c_span, zorder=-10)


plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'compare_Nanog_%s.pdf' % rankFolder), facecolor='w', dpi=200)
plt.show()
