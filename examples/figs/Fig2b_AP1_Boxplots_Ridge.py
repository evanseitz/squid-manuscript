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

# used for Figure 2 (AP-1 attribution error boxplots, mut=0)
# environment: e.g., 'mavenn'


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
    
#gauge = 'wildtype'
#gauge = 'empirical'
gauge = 'hierarchical'
#gauge = 'default'

alphabet = ['A','C','G','T']
alpha = 'dna'
fig_pad = 15
seq_total = 50

avgFolder = 'SQUID_13_AP1_intra_mut0/pad%s' % fig_pad
avgFolder_linear = 'SQUID_13_AP1_intra_mut0_ridge/pad%s' % fig_pad

avgDir = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder)
avgDir_linear = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_linear)

avg_ISM = pd.read_csv(os.path.join(avgDir, 'ADD_A/ADD_%s/avg_ISM_A.csv' % gauge), index_col=0)
avg_sal = pd.read_csv(os.path.join(avgDir, 'ADD_A/ADD_%s/avg_other_A.csv' % gauge), index_col=0)

avg_add_NL = pd.read_csv(os.path.join(avgDir, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_add_L = pd.read_csv(os.path.join(avgDir_linear, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)

tribox_NL = np.load(os.path.join(avgDir, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_L = np.load(os.path.join(avgDir_linear, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()

wtFolder_NL = 'SQUID_13_AP1_intra_mut0'
wtFolder_L = wtFolder_NL#'SQUID_13_AP1_intra_mut0_ridge'

# fill in the following rankA_seqB indices based on the CLI outputs that will print later in this script
### ISM min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank47_seq3179' % wtFolder_NL)
wt_ISM1 = np.load(os.path.join(wtDir, 'attributions_ISM_single.npy'))
wt_ISM1 = squid_utils.fix_gauge(np.array(wt_ISM1), gauge='hierarchical', wt=None)
wt_ISM1 = squid_utils.arr2pd(wt_ISM1, alphabet)

### ISM max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank9_seq17' % wtFolder_NL)
wt_ISM2 = np.load(os.path.join(wtDir, 'attributions_ISM_single.npy'))
wt_ISM2 = squid_utils.fix_gauge(np.array(wt_ISM2), gauge='hierarchical', wt=None)
wt_ISM2 = squid_utils.arr2pd(wt_ISM2, alphabet)

### Saliency min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank31_seq1070' % wtFolder_NL)
wt_sal1 = np.load(os.path.join(wtDir, 'attributions_saliency.npy'))
wt_sal1 = squid_utils.fix_gauge(np.array(wt_sal1), gauge='hierarchical', wt=None)
wt_sal1 = squid_utils.arr2pd(wt_sal1, alphabet)
### Saliency max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank35_seq383' % wtFolder_NL) #rank9_seq17
wt_sal2 = np.load(os.path.join(wtDir, 'attributions_saliency.npy'))
wt_sal2 = squid_utils.fix_gauge(np.array(wt_sal2), gauge='hierarchical', wt=None)
wt_sal2 = squid_utils.arr2pd(wt_sal2, alphabet)

### Additive Ridge min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank3_seq261' % wtFolder_L)
wt_add1_L = pd.read_csv(os.path.join(wtDir, 'ridge_additive.csv'), index_col=0)
wt_add1_L = squid_utils.fix_gauge(np.array(wt_add1_L), gauge='hierarchical', wt=None)
wt_add1_L = squid_utils.arr2pd(wt_add1_L, alphabet)
### Additive Ridge max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank15_seq2554' % wtFolder_L)
wt_add2_L = pd.read_csv(os.path.join(wtDir, 'ridge_additive.csv'), index_col=0)
wt_add2_L = squid_utils.fix_gauge(np.array(wt_add2_L), gauge='hierarchical', wt=None)
wt_add2_L = squid_utils.arr2pd(wt_add2_L, alphabet)

### Additive Nonlinear min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank26_seq1538' % wtFolder_NL)
wt_add1_NL = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
wt_add1_NL = squid_utils.fix_gauge(np.array(wt_add1_NL), gauge='hierarchical', wt=None)
wt_add1_NL = squid_utils.arr2pd(wt_add1_NL, alphabet)
### Additive Nonlinear max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank43_seq3573' % wtFolder_NL)
wt_add2_NL = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
wt_add2_NL = squid_utils.fix_gauge(np.array(wt_add2_NL), gauge='hierarchical', wt=None)
wt_add2_NL = squid_utils.arr2pd(wt_add2_NL, alphabet)


fig = plt.figure(figsize=[15,4])#,constrained_layout=True)

#gs1 = GridSpec(4, 3, left=0.05, right=0.48, wspace=0.1, hspace=0.1)
gs1 = GridSpec(4, 3, left=0.05, right=0.37, wspace=0.1, hspace=0.2)
ax1 = fig.add_subplot(gs1[:4, :3])
#ax1.axes.get_xaxis().set_ticks([])

gs0 = GridSpec(4, 3, left=0.38, right=0.88, wspace=0.1, hspace=0.2) #.38,.68
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

gs0 = GridSpec(4, 3, left=0.49+.055, right=0.99+.055, wspace=0.1, hspace=0.2) #.49, .79
ax6 = fig.add_subplot(gs0[0, 0])
ax6.axes.get_xaxis().set_ticks([])
ax6.axes.get_yaxis().set_ticks([])
ax7 = fig.add_subplot(gs0[1, 0])
ax7.axes.get_xaxis().set_ticks([])
ax7.axes.get_yaxis().set_ticks([])
ax8 = fig.add_subplot(gs0[2, 0])
ax8.axes.get_xaxis().set_ticks([])
ax8.axes.get_yaxis().set_ticks([])
ax9 = fig.add_subplot(gs0[3, 0])
ax9.axes.get_xaxis().set_ticks([])
ax9.axes.get_yaxis().set_ticks([])

gs0 = GridSpec(4, 3, left=0.60+.11, right=1.09+.11, wspace=0.1, hspace=0.2)
ax10 = fig.add_subplot(gs0[0, 0])
ax10.axes.get_xaxis().set_ticks([])
ax10.axes.get_yaxis().set_ticks([])
ax11 = fig.add_subplot(gs0[1, 0])
ax11.axes.get_xaxis().set_ticks([])
ax11.axes.get_yaxis().set_ticks([])
ax12 = fig.add_subplot(gs0[2, 0])
ax12.axes.get_xaxis().set_ticks([])
ax12.axes.get_yaxis().set_ticks([])
ax13 = fig.add_subplot(gs0[3, 0])
ax13.axes.get_xaxis().set_ticks([])
ax13.axes.get_yaxis().set_ticks([])

# =============================================================================
# Box plots
# =============================================================================
ism_boxes = list(tribox_L.values())[0]
sal_boxes = list(tribox_L.values())[1]
add_boxes_L = list(tribox_L.values())[2]
add_boxes_NL = list(tribox_NL.values())[2]

print('ISM min/max index: %s, %s' % (np.argmin(ism_boxes), np.argmax(ism_boxes)))
print('saliency min/max index: %s, %s' % (np.argmin(sal_boxes), np.argmax(sal_boxes)))
print('additive_L min/max index: %s, %s' % (np.argmin(add_boxes_L), np.argmax(add_boxes_L)))
print('additive_NL min/max index: %s, %s' % (np.argmin(add_boxes_NL), np.argmax(add_boxes_NL)))


#A, B, C = -1, 0, 1
A, B, C, D = -1.5, -0.5, 0.5, 1.5
flierprops = dict(marker='>', markeredgecolor='k', markerfacecolor='k', markersize=10, linestyle='none')
ism_plot = ax1.boxplot(ism_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[D], vert=False)
for median in ism_plot['medians']:
    median.set_color('black')
sal_plot = ax1.boxplot(sal_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[C], vert=False)
for median in sal_plot['medians']:
    median.set_color('black')
add_plot_L = ax1.boxplot(add_boxes_L, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[B], vert=False)
for median in add_plot_L['medians']:
    median.set_color('black')

add_plot_NL = ax1.boxplot(add_boxes_NL, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[A], vert=False)
for median in add_plot_NL['medians']:
    median.set_color('black')

s = 8 #scatter scale
a = .5 #scatter alpha
color_ISM = '#377eb8' #blue
color_sal = '#ff7f00' #orange
color_dE = '#e41a1c' #red
color_dL = '#984ea3' #purple
color_add_L = '#f781bf' #pink
color_add_NL = '#4daf4a' #green

ax1.set_xlabel('Error', fontsize=16, labelpad=3)

singles_x = np.random.normal(D, 0.08, size=len(ism_boxes))
ax1.scatter(ism_boxes, singles_x, alpha=a, s=s, c=color_ISM, zorder=-10)
singles_x = np.random.normal(C, 0.08, size=len(sal_boxes))
ax1.scatter(sal_boxes, singles_x, alpha=a, s=s, c=color_sal, zorder=-10)
singles_x = np.random.normal(B, 0.08, size=len(add_boxes_L))
ax1.scatter(add_boxes_L, singles_x, alpha=a, s=s, c=color_add_L, zorder=-10)
singles_x = np.random.normal(A, 0.08, size=len(add_boxes_NL))
ax1.scatter(add_boxes_NL, singles_x, alpha=a, s=s, c=color_add_NL, zorder=-10)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
set_box_color(ism_plot, color_ISM)
set_box_color(sal_plot, color_sal)
set_box_color(add_plot_L, color_add_L)
set_box_color(add_plot_NL, color_add_NL)

ax1.set_xlim(0, ax1.get_xlim()[1]+2.5)
ax1.set_yticklabels(['ISM', 'Saliency', 'Linear', 'Nonlinear'], fontsize=12)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


if 1:
    ax_tops = []
    ax_bots = []
    bar_heights = []
    ax_tops.append([item.get_ydata()[1] for item in ism_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in sal_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in add_plot_L['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in add_plot_NL['caps']][1])
    ax_bots.append([item.get_ydata()[1] for item in ism_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in sal_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in add_plot_L['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in add_plot_NL['caps']][0])

    # add MWU information to plot
    alt = 'less' #{'two-sided', 'less'}
    #combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    combinations = [(A, C), (A, B), (A, D), (B, C), (B, D), (C, D)]
    ax_bottom, ax_top = ax1.get_ylim()
    y_range = ax_top - ax_bottom
    level_idx = 3
    for x1, x2 in combinations:#range(len(comparison_methods)):
        if x1 == A and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(add_boxes_NL, ism_boxes, alternative=alt)
        elif x1 == B and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(add_boxes_NL, sal_boxes, alternative=alt)
        elif x1 == C and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(add_boxes_NL, add_boxes_L, alternative=alt)
        else:
            continue
        #print('MWU statistic (%s–%s): %s' % (x1,x2,mwu_stat))
        print('MWU p-value (%s–%s): %s' % (x1,x2,pval))

        if 0:
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
            ax1.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            text_height = bar_height + (y_range * 0.001)
            ax1.text((x1 + x2) * 0.5, text_height-.2, sig_symbol, ha='center', va='bottom', c='k') #text height needs to substract constant due to PDF saving error
            level_idx -= 1
    if 0:
        ax1.set_ylim(0, ax1.get_ylim()[1]+1.5)

    #for c in list(range(-1, 2)):
        #ax1.text(c, 0 + y_range * 0.02, r'$\it{n} =$ %s' % seq_total, ha='center', size='small')


# =============================================================================
# Attribution maps for individual sequences
# =============================================================================
if 1:
    start, stop = 964, 970
    logo = logomaker.Logo(df=wt_ISM1[start-fig_pad:stop+fig_pad+1],
                        ax=ax2,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold',
                        color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank47_seq3179' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_ISM)


    ax2.set_title('Min Error', fontsize=16, pad=10)

    start, stop = 1038, 1044
    logo = logomaker.Logo(df=wt_sal1[start-fig_pad:stop+fig_pad+1],
                    ax=ax3,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank31_seq1070' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_sal)

    start, stop = 755, 761
    logo = logomaker.Logo(df=wt_add1_L[start-fig_pad:stop+fig_pad+1],
                    ax=ax4,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank3_seq261' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_add_L)

    start, stop = 945, 951
    logo = logomaker.Logo(df=wt_add1_NL[start-fig_pad:stop+fig_pad+1],
                    ax=ax5,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank26_seq1538' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_add_NL)


    ax6.set_title('Max Error', fontsize=16, pad=10)

    start, stop = 1028, 1034
    logo = logomaker.Logo(df=wt_ISM2[start-fig_pad:stop+fig_pad+1],
                    ax=ax6,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank9_seq17' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_ISM)

    start, stop = 1063, 1069 #1028, 1034
    logo = logomaker.Logo(df=wt_sal2[start-fig_pad:stop+fig_pad+1],
                    ax=ax7,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank35_seq383' % wtFolder_NL) #rank9_seq17
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_sal)

    start, stop = 1200, 1206
    logo = logomaker.Logo(df=wt_add2_L[start-fig_pad:stop+fig_pad+1],
                    ax=ax8,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank15_seq2554' % wtFolder_L)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_add_L)

    start, stop = 1118, 1124
    logo = logomaker.Logo(df=wt_add2_NL[start-fig_pad:stop+fig_pad+1],
                    ax=ax9,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    color_scheme='dimgray')
    refDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank43_seq3573' % wtFolder_NL)
    ref_attr = pd.read_csv(os.path.join(refDir, 'attributions_ISM_single.csv'), index_col=0)
    ref_arr = np.where(np.array(ref_attr)!=0., 1, 0)
    ref_seq = squid_utils.oh2seq(ref_arr[start-fig_pad:stop+fig_pad+1], ['A','C','G','T'])
    logo.style_glyphs_in_sequence(sequence=ref_seq, color=color_add_NL)

    # =============================================================================
    # Averaged attribution maps
    # =============================================================================
    ax10.set_title('Average', fontsize=16, pad=10)

    logomaker.Logo(df=avg_ISM,
                    ax=ax10,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_sal,
                    ax=ax11,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_add_L,
                    ax=ax12,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_add_NL,
                    ax=ax13,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')


if 1:
    plt.savefig(os.path.join(pyDir,'boxplot_solo_pad%s_%s.pdf' % (fig_pad, gauge)), facecolor='w', dpi=200)
#plt.tight_layout()
plt.show()


# =============================================================================
# Create standalone legend
# =============================================================================
'''if 0:
    colors = ['white', 'white', 'white']
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(len(colors))]
    labels = [r'$p<0.05$', r'$p<0.01$', r'$p<0.001$']
    legend = plt.legend(handles, labels, loc='center', framealpha=1, frameon=True)
    
    def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    
    export_legend(legend)
    plt.show()'''



