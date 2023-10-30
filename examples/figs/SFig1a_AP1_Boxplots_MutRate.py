import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import pandas as pd
import logomaker
import scipy
from scipy import stats

# used for Figure 1 in Supplementary Material (AP-1 attribution error boxplots, mut=0)


pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils

np.random.seed(0)

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])
    
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
alphabet = ['A','C','G','T']
alpha = 'dna'
fig_pad = 50
seq_total = 50
avgMuts = [1,10,20,30,40,50,60,70,80,90,100]
pos = [1,2,3,4,5,6,7,8,9,10,11]
ylabels = ['1\n0.5$\%$','10\n5$\%$','20\n10$\%$','30\n15$\%$','40\n20$\%$','50\n25$\%$',
           '60\n30$\%$','70\n35$\%$','80\n40$\%$','90\n45$\%$','100\n50$\%$']
color_add_NL = '#4daf4a' #green

boxplots_all = []
attributions_all = np.zeros(shape=(107,4,len(avgMuts)))

# gather average_muts data and consolidate it for figures
for m_idx, m in enumerate(avgMuts):
    avgFolder = 'SQUID_13_AP1_intra_mut0_avgMuts%s_pad%s' % (m,fig_pad)
    avgDir = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single_avgMuts/%s' % avgFolder)

    avg_add = pd.read_csv(os.path.join(avgDir, 'ADD_A/avg_additive_A.csv'), index_col=0)
    tribox = np.load(os.path.join(avgDir, 'stats/compare_boxplot_A_values.npy'), allow_pickle='TRUE').item()
    
    boxplots_all.append(list(tribox.values())[2])
    attributions_all[:,:,m_idx] = np.array(avg_add)

boxplots_all = boxplots_all[::-1]#.reverse()

fig = plt.figure(figsize=[25,10]) #[column width, row width]
gs1 = GridSpec(11, 3, left=0.05, right=0.27, wspace=0.1, hspace=0.2) #.37
ax1 = fig.add_subplot(gs1[:11, :3])
#ax1.axes.get_xaxis().set_ticks([])
gs0 = GridSpec(11, 3, left=0.28, right=1.34, wspace=0.1, hspace=0.2) #.38
ax2 = fig.add_subplot(gs0[0, 0])
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
ax3 = fig.add_subplot(gs0[1, 0])
ax3.axes.get_xaxis().set_ticks([])
ax3.axes.get_yaxis().set_ticks([])
ax4 = fig.add_subplot(gs0[2, 0])
ax4.axes.get_xaxis().set_ticks([])
ax4.axes.get_yaxis().set_ticks([])
ax5 = fig.add_subplot(gs0[3, 0])
ax5.axes.get_xaxis().set_ticks([])
ax5.axes.get_yaxis().set_ticks([])
ax6 = fig.add_subplot(gs0[4, 0])
ax6.axes.get_xaxis().set_ticks([])
ax6.axes.get_yaxis().set_ticks([])
ax7 = fig.add_subplot(gs0[5, 0])
ax7.axes.get_xaxis().set_ticks([])
ax7.axes.get_yaxis().set_ticks([])
ax8 = fig.add_subplot(gs0[6, 0])
ax8.axes.get_xaxis().set_ticks([])
ax8.axes.get_yaxis().set_ticks([])
ax9 = fig.add_subplot(gs0[7, 0])
ax9.axes.get_xaxis().set_ticks([])
ax9.axes.get_yaxis().set_ticks([])
ax10 = fig.add_subplot(gs0[8, 0])
ax10.axes.get_xaxis().set_ticks([])
ax10.axes.get_yaxis().set_ticks([])
ax11 = fig.add_subplot(gs0[9, 0])
ax11.axes.get_xaxis().set_ticks([])
ax11.axes.get_yaxis().set_ticks([])
ax12 = fig.add_subplot(gs0[10, 0])
ax12.axes.get_xaxis().set_ticks([])
ax12.axes.get_yaxis().set_ticks([])

# =============================================================================
# Box plots
# =============================================================================

flierprops = dict(marker='>', markeredgecolor='k', markerfacecolor='k', markersize=10, linestyle='none')
boxplots = ax1.boxplot(boxplots_all, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=pos, vert=False)
for median in boxplots['medians']:
    median.set_color('black')
set_box_color(boxplots, color_add_NL)

s = 8 #scatter scale
a = .5 #scatter alpha
ax1.set_xlabel('Attribution error', fontsize=16, labelpad=3)

for p_idx, p in enumerate(pos):
    np.random.seed(0)
    singles_x = np.random.normal(p, 0.08, size=len(boxplots_all[p_idx]))
    ax1.scatter(boxplots_all[p_idx], singles_x, alpha=a, s=s, c=color_add_NL, zorder=-10)

#ax1.set_xlim(0, ax1.get_xlim()[1]+2.5)
ax1.set_yticklabels(ylabels[::-1], fontsize=12)

# =============================================================================
# Averaged attribution maps
# =============================================================================
ax2.set_title('Average', fontsize=16, pad=10)

logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,0], alphabet),#[50:-50],
                ax=ax2,
                fade_below=.5,
                shade_below=.5,
                width=.9,
                center_values=True,
                font_name='Arial Rounded MT Bold')

if 1:
    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,1], alphabet),
                    ax=ax3,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,2], alphabet),
                    ax=ax4,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,3], alphabet),
                    ax=ax5,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,4], alphabet),
                    ax=ax6,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,5], alphabet),
                    ax=ax7,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,6], alphabet),
                    ax=ax8,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,7], alphabet),
                    ax=ax9,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,8], alphabet),
                    ax=ax10,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,9], alphabet),
                    ax=ax11,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=squid_utils.arr2pd(attributions_all[:,:,10], alphabet),
                    ax=ax12,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'boxplots_avgMuts.pdf'), facecolor='w', dpi=200)
plt.show()








