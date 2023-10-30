import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import pandas as pd
import logomaker
import scipy
from scipy import stats
from scipy.ndimage import uniform_filter1d

import squid.utils as squid_utils

# used for Fig. 4bc and SFig. 3
# see 'Fig4a_PWM_Rankings.py' for generating data used in this script
# turn on switch in 2nd half of script for Supplementary Figure 3
# environment: e.g., 'mavenn'

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)


if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])


#gauge = 'empirical' #{empirical, hierarchical, wildtype, default}
#gauge = 'hierarchical'
gauge = 'wildtype'
#gauge = 'default'

assetDir = os.path.join(pyDir,'Fig4_data/%s' % gauge)

color_ISM = '#377eb8' #blue
color_sal = '#ff7f00' #orange
color_dE = '#e41a1c' #red
color_dL = '#984ea3' #purple
color_add = '#4daf4a' #blue
#cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
N_1d = 20#35
colors = {'0':'#B1B1B1', '1':'#747474', '2':'#444444'}
alphabet = ['A','C','G','T']

fig = plt.figure(figsize=[15,5], constrained_layout=True)

y_c = 6 #add y_c to max
if gauge == 'wildtype':
    ymaxs = [16+y_c, 17+y_c, 18+y_c]
elif gauge == 'empirical': 
    ymaxs = [41, 41, 41]
elif gauge == 'hierarchical':
    ymaxs = [41, 41, 41]
elif gauge == 'default':
    ymaxs = [36, 36, 36]
y_idx = 0

lw = 1 #linewidth for plots

# =============================================================================
# Model 1: ResidualBind-32
# =============================================================================
# box 1 axes setup
gs1 = GridSpec(2, 2, left=0.05, right=0.34, wspace=0, hspace=0.1)
ax1 = fig.add_subplot(gs1[0, 0])
ax1.axes.get_xaxis().set_ticks([])
ax1.set_ylabel('PWM scores', fontsize=16, labelpad=3)
#ax1.yaxis.set_label_coords(-.2,-.05)
ax1.set_title(r'$\it{AP}\textnormal{-}1$ (PC-3)' '\n' r'$\mathrm{\textbf{TGAGTCA}}$', fontsize=10)
ax2 = fig.add_subplot(gs1[1, 0])
ax2.set_ylabel('Error', fontsize=16, labelpad=12)
ax2.set_xlabel('PWM rank', fontsize=16)
ax2.xaxis.set_label_coords(1,-.2)
ax3 = fig.add_subplot(gs1[0, 1])
ax3.axes.get_xaxis().set_ticks([])
ax3.axes.get_yaxis().set_ticks([])
ax3.set_title(r'$\it{IRF1}$ (GM12878)' '\n' r'$\mathrm{\textbf{AA\textmd{N}TGAAAC}}$', fontsize=10)
ax4 = fig.add_subplot(gs1[1, 1])
ax4.axes.get_yaxis().set_ticks([])

ax1.set_ylim(-11,25)
ax1.set_xlim(-10,160)
ax2.set_xlim(-10,160)
ax2.set_ylim(0,ymaxs[y_idx])
ax3.set_ylim(-11,25)
ax3.set_xlim(-10,160)
ax4.set_xlim(-10,160)
ax4.set_ylim(0,ymaxs[y_idx])

scores = np.load(os.path.join(assetDir, 'PWM_scores_GOPHER_13_AP1.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_GOPHER_13_AP1.npy'))
barplot = ax1.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
#ax1.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

scores = np.load(os.path.join(assetDir, 'PWM_scores_GOPHER_7_IRF1-long.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_GOPHER_7_IRF1-long.npy'))
barplot = ax3.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
#ax3.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_GOPHER_13_AP1.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_GOPHER_13_AP1.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_GOPHER_13_AP1.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax2.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax2.plot(ism_smooth, c=color_ISM, linestyle='-')
ax2.plot(other_errors_sort, c=color_sal, alpha=0.2, linewidth=lw)
ax2.plot(other_smooth, c=color_sal, linestyle='-')
ax2.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax2.plot(add_smooth, c=color_add, linestyle='-')

add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_GOPHER_7_IRF1-long.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_GOPHER_7_IRF1-long.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_GOPHER_7_IRF1-long.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax4.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax4.plot(ism_smooth, c=color_ISM, linestyle='-')
ax4.plot(other_errors_sort, c=color_sal, alpha=0.2, linewidth=lw)
ax4.plot(other_smooth, c=color_sal, linestyle='-')
ax4.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax4.plot(add_smooth, c=color_add, linestyle='-')

y_idx += 1

# =============================================================================
# Model 2: DeepSTARR
# =============================================================================
# box 2 axes setup
gs2 = GridSpec(2, 2, left=0.365, right=0.655, wspace=0, hspace=0.1)
ax5 = fig.add_subplot(gs2[0, 0])
ax5.set_ylim(-11,25)
ax5.axes.get_xaxis().set_ticks([])
ax5.axes.get_yaxis().set_ticks([])
ax5.set_title(r'$\it{Dref}$ (hk)' '\n' r'$\mathrm{\textbf{TATCGATA}}$', fontsize=10)
#ax5.set_title(r'$\it{AP}\textnormal{-}1$ (dev)' '\n' r'$\mathrm{\textbf{TGACTCA}}$', fontsize=10)
ax6 = fig.add_subplot(gs2[1, 0])
#ax6.axes.get_yaxis().set_ticks([])
ax6.set_xlabel('PWM rank', fontsize=16)
ax6.xaxis.set_label_coords(1,-.2)
ax7 = fig.add_subplot(gs2[0, 1])
ax7.axes.get_yaxis().set_ticks([])
ax7.set_ylim(-11,25)
ax7.axes.get_xaxis().set_ticks([])
ax7.set_title(r'$\it{Ohler1}$ (hk)' '\n' r'$\mathrm{\textbf{AGTGTGACC}}$', fontsize=10)
ax8 = fig.add_subplot(gs2[1, 1])
ax8.axes.get_yaxis().set_ticks([])

scores = np.load(os.path.join(assetDir, 'PWM_scores_DeepSTARR_DRE.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_DeepSTARR_DRE.npy'))
barplot = ax5.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
#ax5.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

scores = np.load(os.path.join(assetDir, 'PWM_scores_DeepSTARR_Ohler1.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_DeepSTARR_Ohler1.npy'))
barplot = ax7.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
#ax7.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_DeepSTARR_DRE.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_DeepSTARR_DRE.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_DeepSTARR_DRE.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax6.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax6.plot(ism_smooth, c=color_ISM, linestyle='-')
ax6.plot(other_errors_sort, c=color_dE, alpha=0.2, linewidth=lw)
ax6.plot(other_smooth, c=color_dE, linestyle='-')
ax6.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax6.plot(add_smooth, c=color_add, linestyle='-')

add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_DeepSTARR_Ohler1.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_DeepSTARR_Ohler1.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_DeepSTARR_Ohler1.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax8.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax8.plot(ism_smooth, c=color_ISM, linestyle='-')
ax8.plot(other_errors_sort, c=color_dE, alpha=0.2, linewidth=lw)
ax8.plot(other_smooth, c=color_dE, linestyle='-')
ax8.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax8.plot(add_smooth, c=color_add, linestyle='-')

ax5.set_ylim(-11,25)
ax5.set_xlim(-10,160)
ax6.set_xlim(-10,160)
ax6.set_ylim(0,ymaxs[y_idx])
ax7.set_ylim(-11,25)
ax7.set_xlim(-10,160)
ax8.set_xlim(-10,160)
ax8.set_ylim(0,ymaxs[y_idx])

y_idx += 1

# =============================================================================
# Model 3: BPNet
# =============================================================================
# box 3 axes setup
gs3 = GridSpec(2, 2, left=0.68, right=0.97, wspace=0, hspace=0.1)
ax9 = fig.add_subplot(gs3[0, 0])
ax9.set_ylim(-11,25)
ax9.axes.get_xaxis().set_ticks([])
ax9.axes.get_yaxis().set_ticks([])
ax9.set_title(r'$\it{Sox2}$ (Sox2)' '\n' r'$\mathrm{\textbf{GAACAATAG}}$', fontsize=10)
ax10 = fig.add_subplot(gs3[1, 0])
#ax10.axes.get_yaxis().set_ticks([])
ax10.set_xlabel('PWM rank', fontsize=16)
ax10.xaxis.set_label_coords(1,-.2)
ax11 = fig.add_subplot(gs3[0, 1])
ax11.set_ylim(-11,25)
ax11.axes.get_yaxis().set_ticks([])
ax11.axes.get_xaxis().set_ticks([])
#ax11.set_title(r'$\it{Oct4\textnormal{-}Sox2}$ (Oct4)' '\n' r'$\mathrm{\textbf{\textmd{TTNNN}ATGCAAA}}$', fontsize=10)
ax11.set_title(r'$\it{Nanog}$ (Nanog)' '\n' r'$\mathrm{\textbf{AGCCATCAA}}$', fontsize=10)
ax12 = fig.add_subplot(gs3[1, 1])
ax12.axes.get_yaxis().set_ticks([])

scores = np.load(os.path.join(assetDir, 'PWM_scores_BPNet_Sox2.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_BPNet_Sox2.npy'))
barplot = ax9.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
#ax9.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

#scores = np.load(os.path.join(assetDir, 'PWM_scores_BPNet_Oct4-Sox2_N.npy'))
#pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_BPNet_Oct4-Sox2_N.npy'))
scores = np.load(os.path.join(assetDir, 'PWM_scores_BPNet_Nanog.npy'))
pwm_colors = np.load(os.path.join(assetDir, 'PWM_colors_BPNet_Nanog.npy'))
barplot = ax11.bar(np.arange(len(scores)), scores)
for b in range(len(scores)):
    if pwm_colors[b] == '#17becf':
        barplot[b].set_color('#B1B1B1')
    elif pwm_colors[b] == '#1f77b4':
        barplot[b].set_color('#747474')
    elif pwm_colors[b] == '#ff7f0e':
        barplot[b].set_color('#444444')
#ax11.legend(handles, labels, loc='best', prop={'size': 6})#, title='muts')

add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_BPNet_Sox2.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_BPNet_Sox2.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_BPNet_Sox2.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax10.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax10.plot(ism_smooth, c=color_ISM, linestyle='-')
ax10.plot(other_errors_sort, c=color_dL, alpha=0.2, linewidth=lw)
ax10.plot(other_smooth, c=color_dL, linestyle='-')
ax10.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax10.plot(add_smooth, c=color_add, linestyle='-')

#add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_BPNet_Oct4-Sox2_N.npy'))
#ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_BPNet_Oct4-Sox2_N.npy'))
#other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_BPNet_Oct4-Sox2_N.npy'))
add_errors_sort = np.load(os.path.join(assetDir, 'Errors_add_BPNet_Nanog.npy'))
ism_errors_sort = np.load(os.path.join(assetDir, 'Errors_ISM_BPNet_Nanog.npy'))
other_errors_sort = np.load(os.path.join(assetDir, 'Errors_other_BPNet_Nanog.npy'))
ism_smooth = uniform_filter1d(ism_errors_sort, size=N_1d)
other_smooth = uniform_filter1d(other_errors_sort, size=N_1d)
add_smooth = uniform_filter1d(add_errors_sort, size=N_1d)
ax12.plot(ism_errors_sort, c=color_ISM, alpha=0.2, linewidth=lw)
ax12.plot(ism_smooth, c=color_ISM, linestyle='-')
ax12.plot(other_errors_sort, c=color_dL, alpha=0.2, linewidth=lw)
ax12.plot(other_smooth, c=color_dL, linestyle='-')
ax12.plot(add_errors_sort, c=color_add, alpha=0.2, linewidth=lw)
ax12.plot(add_smooth, c=color_add, linestyle='-')

ax9.set_ylim(-11,25)
ax9.set_xlim(-10,160)
ax10.set_xlim(-10,160)
ax10.set_ylim(0,ymaxs[y_idx])
ax11.set_ylim(-11,25)
ax11.set_xlim(-10,160)
ax12.set_xlim(-10,160)
ax12.set_ylim(0,ymaxs[y_idx])

if 1:
    plt.savefig(os.path.join(pyDir,'PWM_compare_models_top_%s.pdf' % gauge), facecolor='w', dpi=200)

if 1:
    plt.show()

    
# =============================================================================
# Supplementary Figure 3
# =============================================================================
if 0:
    model_pad = 50  
    fig_pad = 20

    fig = plt.figure(figsize=[15,3], constrained_layout=True)

    # =============================================================================
    # Model 1: ResidualBind-32
    # =============================================================================
    # box 1 axes setup
    gs4 = GridSpec(5, 3, left=0.05, right=0.51, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs4[0, 0])
    ax1.set_title('ISM', fontsize=10)
    ax2 = fig.add_subplot(gs4[1, 0])
    ax3 = fig.add_subplot(gs4[2, 0])
    ax10 = fig.add_subplot(gs4[3, 0])
    ax13 = fig.add_subplot(gs4[4, 0])

    ax4 = fig.add_subplot(gs4[0, 1])
    ax4.set_title('Saliency', fontsize=10)
    ax5 = fig.add_subplot(gs4[1, 1])
    ax6 = fig.add_subplot(gs4[2, 1])
    ax11 = fig.add_subplot(gs4[3, 1])
    ax14 = fig.add_subplot(gs4[4, 1])

    ax7 = fig.add_subplot(gs4[0, 2])
    ax7.set_title('Additive', fontsize=10)
    ax8 = fig.add_subplot(gs4[1, 2])
    ax9 = fig.add_subplot(gs4[2, 2])
    ax12 = fig.add_subplot(gs4[3, 2])
    ax15 = fig.add_subplot(gs4[4, 2])

    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    ax4.axes.get_xaxis().set_ticks([])
    ax4.axes.get_yaxis().set_ticks([])
    ax5.axes.get_xaxis().set_ticks([])
    ax5.axes.get_yaxis().set_ticks([])
    ax6.axes.get_xaxis().set_ticks([])
    ax6.axes.get_yaxis().set_ticks([])
    ax7.axes.get_xaxis().set_ticks([])
    ax7.axes.get_yaxis().set_ticks([])
    ax8.axes.get_xaxis().set_ticks([])
    ax8.axes.get_yaxis().set_ticks([])
    ax9.axes.get_xaxis().set_ticks([])
    ax9.axes.get_yaxis().set_ticks([])
    ax10.axes.get_xaxis().set_ticks([])
    ax10.axes.get_yaxis().set_ticks([])
    ax11.axes.get_xaxis().set_ticks([])
    ax11.axes.get_yaxis().set_ticks([])
    ax12.axes.get_xaxis().set_ticks([])
    ax12.axes.get_yaxis().set_ticks([])
    ax13.axes.get_xaxis().set_ticks([])
    ax13.axes.get_yaxis().set_ticks([])
    ax14.axes.get_xaxis().set_ticks([])
    ax14.axes.get_yaxis().set_ticks([])
    ax15.axes.get_xaxis().set_ticks([])
    ax15.axes.get_yaxis().set_ticks([])

    surrogate_path = os.path.join(parentDir,'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/') 
    analysis_path = os.path.join(parentDir,'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single') 
    motif_A_name = '7_IRF1-long'
    motif_A = 'AANTGAAAC'

    avg_add_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_additive_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
    avg_ism_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_ISM_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
    avg_other_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_other_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)

    #seq1_start = 921
    #seq1_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank41_seq912/mavenn_additive.csv'), index_col=0)
    #seq1_ism = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank41_seq912/attributions_ISM_single.npy'))
    #seq1_other = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank41_seq912/attributions_saliency.npy'))
    seq1_start = 919
    seq1_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank2_seq2051/mavenn_additive.csv'), index_col=0)
    seq1_ism = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank2_seq2051/attributions_ISM_single.npy'))
    seq1_other = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut0/rank2_seq2051/attributions_saliency.npy'))

    seq2_start = 878
    seq2_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut1/rank9_seq2665/mavenn_additive.csv'), index_col=0)
    seq2_ism = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut1/rank9_seq2665/attributions_ISM_single.npy'))
    seq2_other = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut1/rank9_seq2665/attributions_saliency.npy'))

    seq3_start = 934
    seq3_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank10_seq2541/mavenn_additive.csv'), index_col=0)
    seq3_ism = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank10_seq2541/attributions_ISM_single.npy'))
    seq3_other = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank10_seq2541/attributions_saliency.npy'))

    seq4_start = 487
    seq4_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank39_seq670/mavenn_additive.csv'), index_col=0)
    seq4_ism = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank39_seq670/attributions_ISM_single.npy'))
    seq4_other = np.load(os.path.join(surrogate_path, 'SQUID_7_IRF1-long_intra_mut2/rank39_seq670/attributions_saliency.npy'))


    seq1_ism = squid_utils.arr2pd(seq1_ism, alphabet)
    seq1_other = squid_utils.arr2pd(seq1_other, alphabet)
    seq2_ism = squid_utils.arr2pd(seq2_ism, alphabet)
    seq2_other = squid_utils.arr2pd(seq2_other, alphabet)
    seq3_ism = squid_utils.arr2pd(seq3_ism, alphabet)
    seq3_other = squid_utils.arr2pd(seq3_other, alphabet)
    seq4_ism = squid_utils.arr2pd(seq4_ism, alphabet)
    seq4_other = squid_utils.arr2pd(seq4_other, alphabet)


    logomaker.Logo(df=avg_ism_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax1,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=avg_other_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax4,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=avg_add_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax7,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_ism[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax2,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_other[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax5,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_add[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax8,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_ism[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax3,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_other[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax6,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_add[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax9,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_ism[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax10,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_other[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax11,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_add[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax12,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_ism[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax13,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_other[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax14,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_add[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax15,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')


    # =============================================================================
    # Model 2: DeepSTARR
    # =============================================================================
    # box 1 axes setup
    gs5 = GridSpec(5, 3, left=0.52, right=0.98, wspace=0, hspace=0)

    ax10 = fig.add_subplot(gs5[0, 0])
    ax10.set_title('ISM', fontsize=10)
    ax11 = fig.add_subplot(gs5[1, 0])
    ax12 = fig.add_subplot(gs5[2, 0])
    ax19 = fig.add_subplot(gs5[3, 0])
    ax22 = fig.add_subplot(gs5[4, 0])

    ax13 = fig.add_subplot(gs5[0, 1])
    ax13.set_title('DeepExplainer', fontsize=10)
    ax14 = fig.add_subplot(gs5[1, 1])
    ax15 = fig.add_subplot(gs5[2, 1])
    ax20 = fig.add_subplot(gs5[3, 1])
    ax23 = fig.add_subplot(gs5[4, 1])

    ax16 = fig.add_subplot(gs5[0, 2])
    ax16.set_title('Additive', fontsize=10)
    ax17 = fig.add_subplot(gs5[1, 2])
    ax18 = fig.add_subplot(gs5[2, 2])
    ax21 = fig.add_subplot(gs5[3, 2])
    ax24 = fig.add_subplot(gs5[4, 2])

    ax10.axes.get_xaxis().set_ticks([])
    ax10.axes.get_yaxis().set_ticks([])
    ax11.axes.get_xaxis().set_ticks([])
    ax11.axes.get_yaxis().set_ticks([])
    ax12.axes.get_xaxis().set_ticks([])
    ax12.axes.get_yaxis().set_ticks([])
    ax13.axes.get_xaxis().set_ticks([])
    ax13.axes.get_yaxis().set_ticks([])
    ax14.axes.get_xaxis().set_ticks([])
    ax14.axes.get_yaxis().set_ticks([])
    ax15.axes.get_xaxis().set_ticks([])
    ax15.axes.get_yaxis().set_ticks([])
    ax16.axes.get_xaxis().set_ticks([])
    ax16.axes.get_yaxis().set_ticks([])
    ax17.axes.get_xaxis().set_ticks([])
    ax17.axes.get_yaxis().set_ticks([])
    ax18.axes.get_xaxis().set_ticks([])
    ax18.axes.get_yaxis().set_ticks([])
    ax19.axes.get_xaxis().set_ticks([])
    ax19.axes.get_yaxis().set_ticks([])
    ax20.axes.get_xaxis().set_ticks([])
    ax20.axes.get_yaxis().set_ticks([])
    ax21.axes.get_xaxis().set_ticks([])
    ax21.axes.get_yaxis().set_ticks([])
    ax22.axes.get_xaxis().set_ticks([])
    ax22.axes.get_yaxis().set_ticks([])
    ax23.axes.get_xaxis().set_ticks([])
    ax23.axes.get_yaxis().set_ticks([])
    ax24.axes.get_xaxis().set_ticks([])
    ax24.axes.get_yaxis().set_ticks([])

    surrogate_path = os.path.join(parentDir,'examples_DeepSTARR/c_surrogate_outputs/model_DeepSTARR/')
    analysis_path = os.path.join(parentDir,'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR')
    motif_A_name = 'Ohler1'
    motif_A = 'AGTGTGACC'
    avg_add_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_additive_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
    avg_ism_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_ISM_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
    avg_other_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_other_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)

    seq1_start = 119
    seq1_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut0/rank0_seq22627/mavenn_additive.csv'), index_col=0)
    seq1_ism = np.load(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut0/rank0_seq22627/attributions_ISM_single.npy'))
    seq1_other = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut0/rank0_seq22627/attributions_deepLIFT_hypothetical.csv'), index_col=0)

    seq2_start = 109
    seq2_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut1/rank24_seq1705/mavenn_additive.csv'), index_col=0)
    seq2_ism = np.load(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut1/rank24_seq1705/attributions_ISM_single.npy'))
    seq2_other = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut1/rank24_seq1705/attributions_deepLIFT_hypothetical.csv'), index_col=0)

    seq3_start = 96
    seq3_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank16_seq21783/mavenn_additive.csv'), index_col=0)
    seq3_ism = np.load(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank16_seq21783/attributions_ISM_single.npy'))
    seq3_other = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank16_seq21783/attributions_deepLIFT_hypothetical.csv'), index_col=0)

    seq4_start = 110
    seq4_add = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank35_seq2067/mavenn_additive.csv'), index_col=0)
    seq4_ism = np.load(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank35_seq2067/attributions_ISM_single.npy'))
    seq4_other = pd.read_csv(os.path.join(surrogate_path, 'SQUID_Ohler1_intra_mut2/rank35_seq2067/attributions_deepLIFT_hypothetical.csv'), index_col=0)


    seq1_ism = squid_utils.arr2pd(seq1_ism, alphabet)
    seq2_ism = squid_utils.arr2pd(seq2_ism, alphabet)
    seq3_ism = squid_utils.arr2pd(seq3_ism, alphabet)
    seq4_ism = squid_utils.arr2pd(seq4_ism, alphabet)


    logomaker.Logo(df=avg_ism_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax10,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=avg_other_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax13,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=avg_add_mut0[50-fig_pad:50+len(motif_A)+fig_pad],
                    ax=ax16,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_ism[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax11,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_other[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax14,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq1_add[seq1_start-fig_pad:seq1_start+len(motif_A)+fig_pad],
                    ax=ax17,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_ism[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax12,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_other[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax15,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq2_add[seq2_start-fig_pad:seq2_start+len(motif_A)+fig_pad],
                    ax=ax18,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_ism[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax19,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_other[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax20,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq3_add[seq3_start-fig_pad:seq3_start+len(motif_A)+fig_pad],
                    ax=ax21,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_ism[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax22,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_other[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax23,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    logomaker.Logo(df=seq4_add[seq4_start-fig_pad:seq4_start+len(motif_A)+fig_pad],
                    ax=ax24,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')


    if 1:
        plt.savefig(os.path.join(pyDir,'PWM_compare_models_bot.pdf'), facecolor='w', dpi=200)
    plt.show()




