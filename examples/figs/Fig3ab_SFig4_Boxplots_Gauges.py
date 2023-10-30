import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from scipy import stats

# used for Figure 3ab and Supplementary Figure 4
# environment: e.g., 'conda activate mavenn' – python 3.7.12; pandas 1.3.5

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)

color_ISM = '#377eb8' #blue
color_sal = '#ff7f00' #orange
color_dE = '#e41a1c' #red
color_dL = '#984ea3' #purple
color_add = '#4daf4a' #green

seq_total = 50

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)#
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])

# =============================================================================
# Boxplot comparisons
# =============================================================================

#gauge = 'wildtype'
#gauge = 'empirical'
#gauge = 'hierarchical'
gauge = 'default'

flanks = 50
plot_pvals = False

# set ylimit for plots (+y_c max whisker-bar):
if gauge == 'wildtype':
    y_c = 5
    ymaxs = [18+y_c, 18+y_c, 18+y_c, 16+3, 16+3, 16+3, 19+y_c, 19+y_c, 19+y_c]
elif gauge == 'empirical':
    y_c = 5
    ymaxs = [19+6, 19+6, 19+6, 16+4.5, 16+4.5, 16+4.5, 21+y_c, 21+y_c, 21+y_c]
elif gauge == 'hierarchical':
    y_c = 8
    ymaxs = [26+y_c, 26+y_c, 26+y_c, 23+7, 23+7, 23+7, 26+9, 26+9, 26+9]
elif gauge == 'default':
    y_c = 7.5
    ymaxs = [25+y_c, 25+y_c, 25+y_c, 19+5, 19+5, 19+5, 18+y_c, 18+y_c, 18+y_c]


userDirs = [os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single'), #7_SPI1
            os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single'), #13_AP1
            os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single'), #7_IRF1
            os.path.join(parentDir, 'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR'),
            os.path.join(parentDir, 'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR'),
            os.path.join(parentDir, 'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR'),
            os.path.join(parentDir, 'examples_BPNet/d_outputs_analysis/model_BPNet_OSKN'),
            os.path.join(parentDir, 'examples_BPNet/d_outputs_analysis/model_BPNet_OSKN'),
            os.path.join(parentDir, 'examples_BPNet/d_outputs_analysis/model_BPNet_OSKN'),
            ]


motifs = ['13_AP1', '7_SPI1', '7_IRF1', 'DRE', 'Ohler1', 'AP1', 'Oct4', 'Sox2', 'Nanog']


motifs_fancy = [r'$\it{AP}\textnormal{-}1$ (PC-3)' '\n' r'$\mathrm{\textmd{TGAGTCA}}$',
                r'$\it{SPI1}$ (GM12878)' '\n' r'$\mathrm{\textmd{GGAAGT}}$', 
                r'$\it{IRF1}$ (GM12878)' '\n' r'$\mathrm{\textmd{TGAAAC}}$', 
                r'$\it{Dref}$ (hk)' '\n' r'$\mathrm{\textmd{ATCGAT}}$',
                r'$\it{Ohler1}$ (hk)' '\n' r'$\mathrm{\textmd{AGTGTGACC}}$',
                r'$\it{AP}\textnormal{-}1 (dev)$' '\n' r'$\mathrm{\textmd{TGAGTCA}}$',
                r'$\it{Oct4}$ (Oct4)' '\n' r'$\mathrm{\textmd{TTTGCAT}}$',
                r'$\it{Sox2}$ (Sox2)' '\n' r'$\mathrm{\textmd{GAACAATAG}}$',
                r'$\it{Nanog}$ (Nanog)' '\n' r'$\mathrm{\textmd{AGCCATCAA}}$',
                ]
    
mutations = [0, 0, 0, 0, 0, 0, 0, 0, 0]
other_attr = ['Saliency', 'Saliency', 'Saliency', 'DeepSHAP', 'DeepSHAP', 'DeepSHAP', 'DeepLIFT', 'DeepLIFT', 'DeepLIFT']
models = [0, 0, 0, 1, 1, 1, 2, 2, 2]
model_names = ['ResidualBind-32\n', 'ResidualBind-32\n', 'ResidualBind-32\n \large{(Human cell lines)}',
               'DeepSTARR\n', 'DeepSTARR\n', 'DeepSTARR\n \large{(Drosophila melanogaster S2 cells)}',
               'BPNet\n', 'BPNet\n', 'BPNet\n \large{(Mouse R1 embryonic stem cells)}']

pad = []
tick_total = []
for padding in range(9):
    pad.append(flanks)
    tick_total.append(3)


# =============================================================================
# Run algorithm to generate boxplot
# =============================================================================
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


fig, axs = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=[15,3])

A, B, C = -.25, 0, .25
x1_ticks = []
x2_ticks = []
ax_tops = []
ax_bots = []
bar_heights = []
pos_idx = 1
pos = np.array([0])
motif_idx = 0
mut_idx = 0
for userDir in userDirs:
    mut_boxes = {}
    sal_boxes = {}
    add_boxes = {}
    idx = 0
    x2_ticks.append(pos_idx)
    dataDir = os.path.join(userDir, 'SQUID_%s_intra_mut0/pad%s/stats/stats_%s' % (motifs[motif_idx], pad[motif_idx], gauge))
    tribox = np.load(os.path.join(dataDir, 'compare_boxplot_A_values.npy'), allow_pickle='TRUE').item()
    mut_boxes[0] = list(tribox.values())[0]
    sal_boxes[0] = list(tribox.values())[1]
    add_boxes[0] = list(tribox.values())[2]
    x1_ticks.append(motifs_fancy[motif_idx])
    
    flierprops = dict(marker='^', markeredgecolor='k', markerfacecolor='k', markersize=5, linestyle='none')
    mut_plot = axs[models[motif_idx]].boxplot(mut_boxes.values(), sym='', widths=0.2, showfliers=False, showmeans=True, meanprops=flierprops,
                           positions=pos-.25)
    sal_plot = axs[models[motif_idx]].boxplot(sal_boxes.values(), sym='', widths=0.2, showfliers=False, showmeans=True, meanprops=flierprops,
                           positions=pos)
    add_plot = axs[models[motif_idx]].boxplot(add_boxes.values(), sym='', widths=0.2, showfliers=False, showmeans=True, meanprops=flierprops,
                           positions=pos+.25)
    
    ax_tops.append([item.get_ydata()[1] for item in mut_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in sal_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in add_plot['caps']][1])
    ax_bots.append([item.get_ydata()[1] for item in mut_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in sal_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in add_plot['caps']][0])

    s = 2 #scatter scale
    a = .3 #scatter alpha
    singles_y = list(mut_boxes.values())[0]
    singles_x = np.random.normal(pos-.25, 0.03, size=len(singles_y))
    axs[models[motif_idx]].scatter(singles_x, singles_y, c=color_ISM, alpha=a, s=s)
    
    singles_y = list(sal_boxes.values())[0]
    singles_x = np.random.normal(pos, 0.03, size=len(singles_y))
    if other_attr[motif_idx] == 'Saliency':
        axs[models[motif_idx]].scatter(singles_x, singles_y, c=color_sal, alpha=a, s=s)
    elif other_attr[motif_idx] == 'DeepSHAP':
        axs[models[motif_idx]].scatter(singles_x, singles_y, c=color_dE, alpha=a, s=s)        
    elif other_attr[motif_idx] == 'DeepLIFT':
        axs[models[motif_idx]].scatter(singles_x, singles_y, c=color_dL, alpha=a, s=s)

    singles_y = list(add_boxes.values())[0]
    singles_x = np.random.normal(pos+.25, 0.03, size=len(singles_y))
    axs[models[motif_idx]].scatter(singles_x, singles_y, c=color_add, alpha=a, s=s)

    set_box_color(mut_plot, color_ISM)
    if other_attr[motif_idx] == 'Saliency':
        set_box_color(sal_plot, color_sal)
    elif other_attr[motif_idx] == 'DeepSHAP':
        set_box_color(sal_plot, color_dE)
    elif other_attr[motif_idx] == 'DeepLIFT':
        set_box_color(sal_plot, color_dL)
    set_box_color(add_plot, color_add)
    
    axs[models[motif_idx]].set_title(model_names[motif_idx], fontsize=14)
    
    # add MWU information to plot
    if plot_pvals:
        alt = 'less' #{'two-sided', 'less'}
        #combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
        combinations = [(A, C), (A, B), (B, C)]
        ax_bottom, ax_top = axs[models[motif_idx]].get_ylim()
        if models[motif_idx] == 0:
            y_range = 22 #ax_top - ax_bottom
        elif models[motif_idx] == 1:
            y_range = 22
        elif models[motif_idx] == 2:
            y_range = 22
        level_idx = 3
        for x1, x2 in combinations:#range(len(comparison_methods)):
            if x1 == A and x2 == C:
                mwu_stat, pval = stats.mannwhitneyu(add_boxes[0], mut_boxes[0], alternative=alt)
            if x1 == A and x2 == B:
                continue
                #mwu_stat, pval = stats.mannwhitneyu(sal_boxes[0], mut_boxes[0], alternative=alt)
            elif x1 == B and x2 == C:
                mwu_stat, pval = stats.mannwhitneyu(add_boxes[0], sal_boxes[0], alternative='less')
            #print('MWU statistic (%s–%s): %s' % (x1,x2,mwu_stat))
            #print('MWU p-value (%s–%s): %s' % (x1,x2,pval))
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
            axs[models[motif_idx]].plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            text_height = bar_height + (y_range * 0.001)
            axs[models[motif_idx]].text((x1 + x2) * 0.5, text_height-.7, sig_symbol, ha='center', va='bottom', c='k') #text height needs to substract constant due to PDF saving error
            level_idx -= 1
        A += 1
        B += 1
        C += 1
    else:
        y_range = 22

    if motif_idx != (len(motifs)-1):
        if models[motif_idx] != models[motif_idx+1]:
            axs[models[motif_idx]].set_xticks(np.arange(0,len(x1_ticks)), x1_ticks, rotation=0, fontsize=10)
            axs[models[motif_idx]].yaxis.set_major_locator(MaxNLocator(integer=True))
            #axs[models[motif_idx]].set_ylim(0, max(bar_heights)+min(ax_bots))
            axs[models[motif_idx]].set_ylim(0, ymaxs[motif_idx])
            axs[models[motif_idx]].set_xlim(-.5,2.5)
            x1_ticks = []
            x2_ticks = []
            pos_idx = 0
            pos = np.array([-1])
            ax_tops = []
            ax_bots = []
            bar_heights = []
            A, B, C = -.25, 0, .25
            # annotate sample size below each box
            for c in list(range(0, 3)):
                axs[models[motif_idx]].text(c, 0 + y_range * 0.02, r'$\it{n} =$ %s' % seq_total, ha='center', size='small')
        else:
            axs[models[motif_idx]].axvline(pos_idx-.5, c='k')

    else:
        axs[models[motif_idx]].set_xticks(np.arange(0,len(x1_ticks)), x1_ticks, rotation=0, fontsize=10)
        axs[models[motif_idx]].yaxis.set_major_locator(MaxNLocator(integer=True))
        #axs[models[motif_idx]].set_ylim(0, max(bar_heights)+min(ax_bots)+1)
        axs[models[motif_idx]].set_ylim(0, ymaxs[motif_idx])
        axs[models[motif_idx]].set_xlim(-.5,2.5)
        axs[models[motif_idx]].plot([], c=color_ISM, label='ISM')
        axs[models[motif_idx]].plot([], c=color_sal, label='Saliency')
        axs[models[motif_idx]].plot([], c=color_dE, label='DeepSHAP')
        axs[models[motif_idx]].plot([], c=color_dL, label='DeepLIFT')
        axs[models[motif_idx]].plot([], c=color_add, label='Additive')
        pos_idx = 0
        pos = np.array([-1])
        ax_tops = []
        ax_bots = []
        bar_heights = []
        A, B, C = -.25, 0, .25
        # annotate sample size below each box
        for c in list(range(0, 3)):
            axs[models[motif_idx]].text(c, 0 + y_range * 0.02, r'$\it{n} =$ %s' % seq_total, ha='center', size='small')

    mut_idx += 1
    motif_idx += 1
    pos_idx += 1
    pos += 1

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
#axs[0].set_ylabel(r'$d({\theta}_{i} - {\rm I\!E}[\Theta])$', fontsize=14, labelpad=14)
axs[0].set_ylabel('Error', fontsize=14, labelpad=14)

plt.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir,'compare_boxplots_%s_pad%s_pvals%s.pdf' % (gauge, flanks, plot_pvals)), facecolor='w', dpi=200)
plt.show()
plt.close()


color_ISM = '#377eb8' #blue
color_sal = '#ff7f00' #orange
color_dE = '#e41a1c' #red
color_dL = '#984ea3' #purple
color_add = '#4daf4a' #blue

# =============================================================================
# Create standalone legend
# =============================================================================
if 0:
    colors = [color_ISM, color_sal, color_dE, color_dL, color_add, 'white', 'white', 'white']
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(len(colors))]
    labels = ['ISM', 'Saliency', 'DeepSHAP   ', 'DeepLIFT', 'Additive', r'$p<0.05$', r'$p<0.01$', r'$p<0.001$']
    legend = plt.legend(handles, labels, loc='center', framealpha=1, frameon=True)
    
    def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=200, bbox_inches=bbox)
    
    export_legend(legend)
    plt.show()
    

# =============================================================================
# Boxplot means
# =============================================================================

if 1: #update per dataset
    # set ylimit for plots (+y_c max whisker-bar):
    if gauge == 'empirical': #for all, ymax = max+2.5
        y_c = 2.5
        ymaxs = [17+y_c, 15+y_c, 19+y_c] 
    elif gauge == 'hierarchical':
        y_c = 3.5
        ymaxs = [24+y_c, 22+y_c, 26+y_c] 
    elif gauge == 'wildtype':
        y_c = 2.5
        ymaxs = [16+y_c, 13+y_c, 17+y_c]
    elif gauge == 'default':
        ymaxs = [24+3.5, 15+3.25, 18+3.5] 

    #############
    # ResBind32 #
    #############
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[5,1.5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    
    x_pad = [0,5,10,15,20,30,40,50,60,70,80,90,100]

    y_idx = 0
    plot_idx = 0

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif1_ISM = avgFlanks[0,x_pad]
    y_motif1_sal = avgFlanks[1,x_pad]
    y_motif1_add = avgFlanks[2,x_pad]
    pticks_motif1 = []
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif1.append('')
        elif p <= 0.001:
            pticks_motif1.append('***')
        elif .001 < p <= .01:
            pticks_motif1.append('**')
        elif .01 < p <= .05:
            pticks_motif1.append('*')
        else:
            pticks_motif1.append('–––')
    plot_idx += 1

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif2_ISM = avgFlanks[0,x_pad]
    y_motif2_sal = avgFlanks[1,x_pad]
    y_motif2_add = avgFlanks[2,x_pad]
    pticks_motif2 = []    
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif2.append('')
        elif p <= 0.001:
            pticks_motif2.append('***')
        elif .001 < p <= .01:
            pticks_motif2.append('**')
        elif .01 < p <= .05:
            pticks_motif2.append('*')
        else:
            pticks_motif2.append('–––')
    plot_idx += 1

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif3_ISM = avgFlanks[0,x_pad]
    y_motif3_sal = avgFlanks[1,x_pad]
    y_motif3_add = avgFlanks[2,x_pad]
    pticks_motif3 = []    
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif3.append('')
        elif p <= 0.001:
            pticks_motif3.append('***')
        elif .001 < p <= .01:
            pticks_motif3.append('**')
        elif .01 < p <= .05:
            pticks_motif3.append('*')
        else:
            pticks_motif3.append('–––')
    plot_idx += 1
    
    ax1,ax2,ax3 = 0,1,2
    
    # motif_1 plot
    axs[ax1].plot(x_pad, y_motif1_ISM, c=color_ISM, zorder=-5)
    axs[ax1].plot(x_pad, y_motif1_sal, c=color_sal, zorder=-4)
    axs[ax1].plot(x_pad, y_motif1_add, c=color_add)
    axs[ax1].scatter(x_pad, y_motif1_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax1].scatter(x_pad, y_motif1_sal, c=color_sal, s=7.5, zorder=-4)
    axs[ax1].scatter(x_pad, y_motif1_add, c=color_add, s=7.5)
    axs[ax1].set_ylim(0,ymaxs[y_idx])
    axs[ax1].set_xlim(-5,x_pad[-1]+10)
    axs[ax1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if ax1 != 2:
        xticks = axs[ax1].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax1].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax1].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif1, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)

    # motif_2 plot
    axs[ax2].plot(x_pad, y_motif2_ISM, c=color_ISM, zorder=-5)
    axs[ax2].plot(x_pad, y_motif2_sal, c=color_sal,zorder=-4)
    axs[ax2].plot(x_pad, y_motif2_add, c=color_add)
    axs[ax2].scatter(x_pad, y_motif2_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax2].scatter(x_pad, y_motif2_sal, c=color_sal, s=7.5, zorder=-4)
    axs[ax2].scatter(x_pad, y_motif2_add, c=color_add, s=7.5)
    axs[ax2].set_ylim(0,ymaxs[y_idx])
    axs[ax2].set_xlim(-5,x_pad[-1]+10)
    if ax2 != 2:
        xticks = axs[ax2].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax2].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax2].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif2, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
    
    # motif_3 plot
    axs[ax3].plot(x_pad, y_motif3_ISM, c=color_ISM, zorder=-5)
    axs[ax3].plot(x_pad, y_motif3_sal, c=color_sal,zorder=-4)
    axs[ax3].plot(x_pad, y_motif3_add, c=color_add)
    axs[ax3].scatter(x_pad, y_motif3_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax3].scatter(x_pad, y_motif3_sal, c=color_sal, s=7.5, zorder=-4)
    axs[ax3].scatter(x_pad, y_motif3_add, c=color_add, s=7.5) 
    axs[ax3].set_ylim(0,ymaxs[y_idx])
    axs[ax3].set_xlim(-5,x_pad[-1]+10)
    if ax3 != 2:
        xticks = axs[ax3].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax3].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax3].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif3, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
        
    [l.set_visible(False) for (i,l) in enumerate(axs[0].yaxis.get_ticklabels()) if i % 3 != 0] # keep every nth label
    
    #axs[ax1].set_ylabel('Mean of Error', fontsize=14, labelpad=14)
       
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'compare_boxplot1_means_%s.pdf' % gauge), facecolor='w', dpi=200)
    plt.show()

    y_idx += 1

    #############
    # DeepSTARR #
    #############

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[5,1.5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif1_ISM = avgFlanks[0,x_pad]
    y_motif1_dL = avgFlanks[1,x_pad]
    y_motif1_add = avgFlanks[2,x_pad]

    x_pad_short1 = []
    pticks_motif1 = []    
    for p_idx, p in enumerate(avgFlanksPval[2,x_pad]):
        if p == 0.:
            pticks_motif1.append('')
        elif p <= 0.001:
            pticks_motif1.append('***')
            x_pad_short1.append(x_pad[p_idx])
        elif .001 < p <= .01:
            pticks_motif1.append('**')
            x_pad_short1.append(x_pad[p_idx])
        elif .01 < p <= .05:
            pticks_motif1.append('*')
            x_pad_short1.append(x_pad[p_idx])
        else:
            pticks_motif1.append('–––')
            x_pad_short1.append(x_pad[p_idx])
    plot_idx += 1

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif2_ISM = avgFlanks[0,x_pad]
    y_motif2_dL = avgFlanks[1,x_pad]
    y_motif2_add = avgFlanks[2,x_pad]
    x_pad_short2 = []
    pticks_motif2 = []    
    for p_idx, p in enumerate(avgFlanksPval[2,x_pad]):
        if p == 0.:
            pticks_motif2.append('')
        elif p <= 0.001:
            pticks_motif2.append('***')
            x_pad_short2.append(x_pad[p_idx])
        elif .001 < p <= .01:
            pticks_motif2.append('**')
            x_pad_short2.append(x_pad[p_idx])
        elif .01 < p <= .05:
            pticks_motif2.append('*')
            x_pad_short2.append(x_pad[p_idx])
        else:
            pticks_motif2.append('–––')
            x_pad_short2.append(x_pad[p_idx])
    plot_idx += 1
    
    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif3_ISM = avgFlanks[0,x_pad]
    y_motif3_dL = avgFlanks[1,x_pad]
    y_motif3_add = avgFlanks[2,x_pad]
    x_pad_short3 = []
    pticks_motif3 = []
    for p_idx, p in enumerate(avgFlanksPval[2,x_pad]):
        if p == 0.:
            pticks_motif3.append('')
        elif p <= 0.001:
            pticks_motif3.append('***')
            x_pad_short3.append(x_pad[p_idx])
        elif .001 < p <= .01:
            pticks_motif3.append('**')
            x_pad_short3.append(x_pad[p_idx])
        elif .01 < p <= .05:
            pticks_motif3.append('*')
            x_pad_short3.append(x_pad[p_idx])
        else:
            pticks_motif3.append('–––')
            x_pad_short3.append(x_pad[p_idx])
    plot_idx += 1

    ax1,ax2,ax3 = 0,1,2
    
    # motif_1 plot
    axs[ax1].plot(x_pad_short1, y_motif1_ISM[:len(x_pad_short1)], c=color_ISM, zorder=-5)
    axs[ax1].plot(x_pad_short1, y_motif1_dL[:len(x_pad_short1)], c=color_dE, zorder=-4)
    axs[ax1].plot(x_pad_short1, y_motif1_add[:len(x_pad_short1)], c=color_add)
    axs[ax1].scatter(x_pad_short1, y_motif1_ISM[:len(x_pad_short1)], c=color_ISM, s=7.5, zorder=-5)
    axs[ax1].scatter(x_pad_short1, y_motif1_dL[:len(x_pad_short1)], c=color_dE, s=7.5, zorder=-4)
    axs[ax1].scatter(x_pad_short1, y_motif1_add[:len(x_pad_short1)], c=color_add, s=7.5)
    axs[ax1].set_ylim(0,ymaxs[y_idx])
    axs[ax1].set_xlim(-5,x_pad[-1]+10)
    axs[ax1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if ax1 != 2:
        xticks = axs[ax1].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax1].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax1].twiny()
    temp_ax.set_xticks(x_pad_short1)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif1[:len(x_pad_short1)], rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
    
    # motif_2 plot
    axs[ax2].plot(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_ISM[:len(x_pad_short2)], c=color_ISM, zorder=-5)
    axs[ax2].plot(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_dL[:len(x_pad_short2)], c=color_dE, zorder=-4)
    axs[ax2].plot(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_add[:len(x_pad_short2)], c=color_add)
    axs[ax2].scatter(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_ISM[:len(x_pad_short2)], c=color_ISM, s=7.5, zorder=-5)
    axs[ax2].scatter(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_dL[:len(x_pad_short2)], c=color_dE, s=7.5, zorder=-4)
    axs[ax2].scatter(x_pad_short2[0:len(y_motif2_ISM)], y_motif2_add[:len(x_pad_short2)], c=color_add, s=7.5)
    axs[ax2].set_ylim(0,ymaxs[y_idx])
    axs[ax2].set_xlim(-5,x_pad[-1]+10)
    if ax2 != 2:
        xticks = axs[ax2].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax2].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax2].twiny()
    temp_ax.set_xticks(x_pad_short2)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif2[:len(x_pad_short2)], rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
    
    # motif_3 plot
    axs[ax3].plot(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_ISM[:len(x_pad_short3)], c=color_ISM, zorder=-5)
    axs[ax3].plot(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_dL[:len(x_pad_short3)], c=color_dE, zorder=-4)
    axs[ax3].plot(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_add[:len(x_pad_short3)], c=color_add)
    axs[ax3].scatter(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_ISM[:len(x_pad_short3)], c=color_ISM, s=7.5, zorder=-5)
    axs[ax3].scatter(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_dL[:len(x_pad_short3)], c=color_dE, s=7.5, zorder=-4)
    axs[ax3].scatter(x_pad_short3[0:len(y_motif3_ISM)], y_motif3_add[:len(x_pad_short3)], c=color_add, s=7.5) 
    axs[ax3].set_ylim(0,ymaxs[y_idx])
    axs[ax3].set_xlim(-5,x_pad[-1]+10)
    if ax3 != 2:
        xticks = axs[ax3].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax3].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax3].twiny()
    temp_ax.set_xticks(x_pad_short3)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif3[:len(x_pad_short3)], rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
      
    [l.set_visible(False) for (i,l) in enumerate(axs[0].yaxis.get_ticklabels()) if i % 3 != 0] # keep every nth label
       
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'compare_boxplot2_means_%s.pdf' % gauge), facecolor='w', dpi=200)
    plt.show()

    y_idx += 1

    #############
    #   BPNet   #
    #############
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[5,1.5], sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    
    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif1_ISM = avgFlanks[0,x_pad]
    y_motif1_dL = avgFlanks[1,x_pad]
    y_motif1_add = avgFlanks[2,x_pad]
    pticks_motif1 = []    
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif1.append('')
        elif p <= 0.001:
            pticks_motif1.append('***')
        elif .001 < p <= .01:
            pticks_motif1.append('**')
        elif .01 < p <= .05:
            pticks_motif1.append('*')
        else:
            pticks_motif1.append('–––')
    plot_idx += 1
            
    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif2_ISM = avgFlanks[0,x_pad]
    y_motif2_dL = avgFlanks[1,x_pad]
    y_motif2_add = avgFlanks[2,x_pad]
    pticks_motif2 = []    
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif2.append('')
        elif p <= 0.001:
            pticks_motif2.append('***')
        elif .001 < p <= .01:
            pticks_motif2.append('**')
        elif .01 < p <= .05:
            pticks_motif2.append('*')
        else:
            pticks_motif2.append('–––')
    plot_idx += 1

    avgFlanks = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s.npy' % (motifs[plot_idx], gauge)))
    avgFlanksPval = np.load(os.path.join(userDirs[plot_idx], 'SQUID_%s_intra_mut0/errors_%s_Pvals.npy' % (motifs[plot_idx], gauge)))
    y_motif3_ISM = avgFlanks[0,x_pad]
    y_motif3_dL = avgFlanks[1,x_pad]
    y_motif3_add = avgFlanks[2,x_pad]
    pticks_motif3 = []    
    for p in avgFlanksPval[2,x_pad]:
        if p == 0.:
            pticks_motif3.append('')
        elif p <= 0.001:
            pticks_motif3.append('***')
        elif .001 < p <= .01:
            pticks_motif3.append('**')
        elif .01 < p <= .05:
            pticks_motif3.append('*')
        else:
            pticks_motif3.append('–––')
    plot_idx += 1
    
    ax1,ax2,ax3 = 0,1,2
    
    # motif_1 plot
    axs[ax1].plot(x_pad, y_motif1_ISM, c=color_ISM, zorder=-5)
    axs[ax1].plot(x_pad, y_motif1_dL, c=color_dL, zorder=-4)
    axs[ax1].plot(x_pad, y_motif1_add, c=color_add)
    axs[ax1].scatter(x_pad, y_motif1_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax1].scatter(x_pad, y_motif1_dL, c=color_dL, s=7.5, zorder=-4)
    axs[ax1].scatter(x_pad, y_motif1_add, c=color_add, s=7.5)
    axs[ax1].set_ylim(0,ymaxs[y_idx])
    axs[ax1].set_xlim(-5,x_pad[-1]+10)
    axs[ax1].yaxis.set_major_locator(MaxNLocator(integer=True))
    if ax1 != 2:
        xticks = axs[ax1].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax1].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax1].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif1, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)

    # motif_2 plot
    axs[ax2].plot(x_pad, y_motif2_ISM, c=color_ISM, zorder=-5)
    axs[ax2].plot(x_pad, y_motif2_dL, c=color_dL, zorder=-4)
    axs[ax2].plot(x_pad, y_motif2_add, c=color_add)
    axs[ax2].scatter(x_pad, y_motif2_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax2].scatter(x_pad, y_motif2_dL, c=color_dL, s=7.5, zorder=-4)
    axs[ax2].scatter(x_pad, y_motif2_add, c=color_add, s=7.5) 
    axs[ax2].set_ylim(0,ymaxs[y_idx])
    axs[ax2].set_xlim(-5,x_pad[-1]+10)
    if ax2 != 2:
        xticks = axs[ax2].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax2].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax2].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif2, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
    
    # motif_3 plot
    axs[ax3].plot(x_pad, y_motif3_ISM, c=color_ISM, label='0.5', zorder=-5)
    axs[ax3].plot(x_pad, y_motif3_dL, c=color_dL, label='0.2', zorder=-4)
    axs[ax3].plot(x_pad, y_motif3_add, c=color_add, label='0.1')
    axs[ax3].scatter(x_pad, y_motif3_ISM, c=color_ISM, s=7.5, zorder=-5)
    axs[ax3].scatter(x_pad, y_motif3_dL, c=color_dL, s=7.5, zorder=-4)
    axs[ax3].scatter(x_pad, y_motif3_add, c=color_add, s=7.5)
    axs[ax3].set_ylim(0,ymaxs[y_idx])
    axs[ax3].set_xlim(-5,x_pad[-1]+10)
    if ax3 != 2:
        xticks = axs[ax3].xaxis.get_major_ticks()
        xticks[-1].set_visible(False)
    axs[ax3].axvline(flanks, linestyle='--', linewidth=1, c='gray', zorder=-10)
    # p-value ticks
    temp_ax = axs[ax3].twiny()
    temp_ax.set_xticks(x_pad)
    temp_ax.set_xlim(0,x_pad[-1]+10)
    temp_ax.set_xticklabels(pticks_motif3, rotation=90, fontsize=8, ha='left') #error in generating PDF, needs ha='left'
    temp_ax.tick_params(axis=u'both', which=u'both',length=0)
    temp_ax.tick_params(axis='x', which='major', pad=1)
        
    [l.set_visible(False) for (i,l) in enumerate(axs[0].yaxis.get_ticklabels()) if i % 3 != 0] # keep every nth label
    
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'compare_boxplot3_means_%s.pdf' % gauge), facecolor='w', dpi=200)
    plt.show()