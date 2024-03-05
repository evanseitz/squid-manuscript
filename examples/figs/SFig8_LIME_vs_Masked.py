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

np.random.seed(2)

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


avgFolder_k10 = 'SQUID_13_AP1_intra_mut0_lime_k10/pad%s' % fig_pad
avgFolder_k10GE = 'SQUID_13_AP1_intra_mut0_lime_k10GE/pad%s' % fig_pad
avgFolder_k20 = 'SQUID_13_AP1_intra_mut0_lime_k20/pad%s' % fig_pad
avgFolder_k20GE = 'SQUID_13_AP1_intra_mut0_lime_k20GE/pad%s' % fig_pad
avgFolder_k50 = 'SQUID_13_AP1_intra_mut0_lime_k50/pad%s' % fig_pad
avgFolder_k50GE = 'SQUID_13_AP1_intra_mut0_lime_k50GE/pad%s' % fig_pad
avgFolder_k100 = 'SQUID_13_AP1_intra_mut0_lime_k100/pad%s' % fig_pad
avgFolder_k100GE = 'SQUID_13_AP1_intra_mut0_lime_k100GE/pad%s' % fig_pad

avgDir_k10 = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k10)
avgDir_k10GE = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k10GE)
avgDir_k20 = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k20)
avgDir_k20GE = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k20GE)
avgDir_k50 = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k50)
avgDir_k50GE = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k50GE)
avgDir_k100 = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k100)
avgDir_k100GE = os.path.join(parentDir, 'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single/%s' % avgFolder_k100GE)

avg_k10 = pd.read_csv(os.path.join(avgDir_k10, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k10GE = pd.read_csv(os.path.join(avgDir_k10GE, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k20 = pd.read_csv(os.path.join(avgDir_k20, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k20GE = pd.read_csv(os.path.join(avgDir_k20GE, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k50 = pd.read_csv(os.path.join(avgDir_k50, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k50GE = pd.read_csv(os.path.join(avgDir_k50GE, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k100 = pd.read_csv(os.path.join(avgDir_k100, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)
avg_k100GE = pd.read_csv(os.path.join(avgDir_k100GE, 'ADD_A/ADD_%s/avg_additive_A.csv' % gauge), index_col=0)

tribox_k10 = np.load(os.path.join(avgDir_k10, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k10GE = np.load(os.path.join(avgDir_k10GE, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k20 = np.load(os.path.join(avgDir_k20, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k20GE = np.load(os.path.join(avgDir_k20GE, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k50 = np.load(os.path.join(avgDir_k50, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k50GE = np.load(os.path.join(avgDir_k50GE, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k100 = np.load(os.path.join(avgDir_k100, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()
tribox_k100GE = np.load(os.path.join(avgDir_k100GE, 'stats/stats_%s/compare_boxplot_A_values.npy' % gauge), allow_pickle='TRUE').item()

wtFolder = 'SQUID_13_AP1_intra_mut0'

uniform_min = 'rank26_seq1538'
uniform_max = 'rank14_seq781'

# fill in the following rankA_seqB indices based on the CLI outputs that will print later in this script
### k10 min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank47_seq3179
wt_k10_1 = pd.read_csv(os.path.join(wtDir, 'lime_k10_additive.csv'), index_col=0)
wt_k10_1 = squid_utils.fix_gauge(np.array(wt_k10_1), gauge='hierarchical', wt=None)
wt_k10_1 = squid_utils.arr2pd(wt_k10_1, alphabet)
### k10 max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank9_seq17
wt_k10_2 = pd.read_csv(os.path.join(wtDir, 'lime_k10_additive.csv'), index_col=0)
wt_k10_2 = squid_utils.fix_gauge(np.array(wt_k10_2), gauge='hierarchical', wt=None)
wt_k10_2 = squid_utils.arr2pd(wt_k10_2, alphabet)

### k10GE min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank31_seq1070
wt_k10GE_1 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k10_1)==0, 0, 1)    
    for i in range(np.array(wt_k10_1).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k10GE_1 = np.array(wt_k10GE_1)*zeros_index
wt_k10GE_1 = squid_utils.fix_gauge(np.array(wt_k10GE_1), gauge='hierarchical', wt=None)
wt_k10GE_1 = squid_utils.arr2pd(wt_k10GE_1, alphabet)
### k10GE max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank35_seq383
wt_k10GE_2 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k10_2)==0, 0, 1)    
    for i in range(np.array(wt_k10_2).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k10GE_2 = np.array(wt_k10GE_2)*zeros_index
wt_k10GE_2 = squid_utils.fix_gauge(np.array(wt_k10GE_2), gauge='hierarchical', wt=None)
wt_k10GE_2 = squid_utils.arr2pd(wt_k10GE_2, alphabet)

### k20 min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank31_seq1070
wt_k20_1 = pd.read_csv(os.path.join(wtDir, 'lime_k20_additive.csv'), index_col=0)
wt_k20_1 = squid_utils.fix_gauge(np.array(wt_k20_1), gauge='hierarchical', wt=None)
wt_k20_1 = squid_utils.arr2pd(wt_k20_1, alphabet)
### k20 max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank35_seq383
wt_k20_2 = pd.read_csv(os.path.join(wtDir, 'lime_k20_additive.csv'), index_col=0)
wt_k20_2 = squid_utils.fix_gauge(np.array(wt_k20_2), gauge='hierarchical', wt=None)
wt_k20_2 = squid_utils.arr2pd(wt_k20_2, alphabet)

### k20GE min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank31_seq1070
wt_k20GE_1 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k20_1)==0, 0, 1)    
    for i in range(np.array(wt_k20_1).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k20GE_1 = np.array(wt_k20GE_1)*zeros_index
wt_k20GE_1 = squid_utils.fix_gauge(np.array(wt_k20GE_1), gauge='hierarchical', wt=None)
wt_k20GE_1 = squid_utils.arr2pd(wt_k20GE_1, alphabet)
### k20GE max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank35_seq383
wt_k20GE_2 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k20_2)==0, 0, 1)    
    for i in range(np.array(wt_k20_2).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k20GE_2 = np.array(wt_k20GE_2)*zeros_index
wt_k20GE_2 = squid_utils.fix_gauge(np.array(wt_k20GE_2), gauge='hierarchical', wt=None)
wt_k20GE_2 = squid_utils.arr2pd(wt_k20GE_2, alphabet)

### k50 min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank3_seq261
wt_k50_1 = pd.read_csv(os.path.join(wtDir, 'lime_k50_additive.csv'), index_col=0)
wt_k50_1 = squid_utils.fix_gauge(np.array(wt_k50_1), gauge='hierarchical', wt=None)
wt_k50_1 = squid_utils.arr2pd(wt_k50_1, alphabet)
### k50 max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank15_seq2554
wt_k50_2 = pd.read_csv(os.path.join(wtDir, 'lime_k50_additive.csv'), index_col=0)
wt_k50_2 = squid_utils.fix_gauge(np.array(wt_k50_2), gauge='hierarchical', wt=None)
wt_k50_2 = squid_utils.arr2pd(wt_k50_2, alphabet)

### k50GE min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank31_seq1070
wt_k50GE_1 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k50_1)==0, 0, 1)    
    for i in range(np.array(wt_k50_1).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k50GE_1 = np.array(wt_k50GE_1)*zeros_index
wt_k50GE_1 = squid_utils.fix_gauge(np.array(wt_k50GE_1), gauge='hierarchical', wt=None)
wt_k50GE_1 = squid_utils.arr2pd(wt_k50GE_1, alphabet)
### k50GE max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank35_seq383
wt_k50GE_2 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k50_2)==0, 0, 1)    
    for i in range(np.array(wt_k50_2).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k50GE_2 = np.array(wt_k50GE_2)*zeros_index
wt_k50GE_2 = squid_utils.fix_gauge(np.array(wt_k50GE_2), gauge='hierarchical', wt=None)
wt_k50GE_2 = squid_utils.arr2pd(wt_k50GE_2, alphabet)

### k100 min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank26_seq1538
wt_k100_1 = pd.read_csv(os.path.join(wtDir, 'lime_k100_additive.csv'), index_col=0)
wt_k100_1 = squid_utils.fix_gauge(np.array(wt_k100_1), gauge='hierarchical', wt=None)
wt_k100_1 = squid_utils.arr2pd(wt_k100_1, alphabet)
### k100 max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank43_seq3573
wt_k100_2 = pd.read_csv(os.path.join(wtDir, 'lime_k100_additive.csv'), index_col=0)
wt_k100_2 = squid_utils.fix_gauge(np.array(wt_k100_2), gauge='hierarchical', wt=None)
wt_k100_2 = squid_utils.arr2pd(wt_k100_2, alphabet)

### k100GE min
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_min)) #rank31_seq1070
wt_k100GE_1 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k100_1)==0, 0, 1)    
    for i in range(np.array(wt_k100_1).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k100GE_1 = np.array(wt_k100GE_1)*zeros_index
wt_k100GE_1 = squid_utils.fix_gauge(np.array(wt_k100GE_1), gauge='hierarchical', wt=None)
wt_k100GE_1 = squid_utils.arr2pd(wt_k100GE_1, alphabet)
### k100GE max
wtDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/%s' % (wtFolder, uniform_max)) #rank35_seq383
wt_k100GE_2 = pd.read_csv(os.path.join(wtDir, 'logo_additive.csv'), index_col=0)
if 1:
    zeros_index = np.where(np.array(wt_k100_2)==0, 0, 1)    
    for i in range(np.array(wt_k100_2).shape[0]):
        if zeros_index[i,0] != 0 or zeros_index[i,1] != 0 or zeros_index[i,2] != 0 or zeros_index[i,3] != 0:
            zeros_index[i,:] = 1
    wt_k100GE_2 = np.array(wt_k100GE_2)*zeros_index
wt_k100GE_2 = squid_utils.fix_gauge(np.array(wt_k100GE_2), gauge='hierarchical', wt=None)
wt_k100GE_2 = squid_utils.arr2pd(wt_k100GE_2, alphabet)


fig = plt.figure(figsize=[15,8])#,constrained_layout=True) [15,4]

#gs1 = GridSpec(4, 3, left=0.05, right=0.48, wspace=0.1, hspace=0.1)
gs1 = GridSpec(8, 3, left=0.05, right=0.37, wspace=0.1, hspace=0.2)
ax1 = fig.add_subplot(gs1[:8, :3])
#ax1.axes.get_xaxis().set_ticks([])

gs0 = GridSpec(8, 3, left=0.38, right=0.88, wspace=0.1, hspace=0.2) #.38,.68
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


gs0 = GridSpec(8, 3, left=0.49+.055, right=0.99+.055, wspace=0.1, hspace=0.2) #.49, .79
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
ax14 = fig.add_subplot(gs0[4, 0])
ax14.axes.get_xaxis().set_ticks([])
ax14.axes.get_yaxis().set_ticks([])
ax15 = fig.add_subplot(gs0[5, 0])
ax15.axes.get_xaxis().set_ticks([])
ax15.axes.get_yaxis().set_ticks([])
ax16 = fig.add_subplot(gs0[6, 0])
ax16.axes.get_xaxis().set_ticks([])
ax16.axes.get_yaxis().set_ticks([])
ax17 = fig.add_subplot(gs0[7, 0])
ax17.axes.get_xaxis().set_ticks([])
ax17.axes.get_yaxis().set_ticks([])

gs0 = GridSpec(8, 3, left=0.60+.11, right=1.09+.11, wspace=0.1, hspace=0.2)
ax18 = fig.add_subplot(gs0[0, 0])
ax18.axes.get_xaxis().set_ticks([])
ax18.axes.get_yaxis().set_ticks([])
ax19 = fig.add_subplot(gs0[1, 0])
ax19.axes.get_xaxis().set_ticks([])
ax19.axes.get_yaxis().set_ticks([])
ax20 = fig.add_subplot(gs0[2, 0])
ax20.axes.get_xaxis().set_ticks([])
ax20.axes.get_yaxis().set_ticks([])
ax21 = fig.add_subplot(gs0[3, 0])
ax21.axes.get_xaxis().set_ticks([])
ax21.axes.get_yaxis().set_ticks([])
ax22 = fig.add_subplot(gs0[4, 0])
ax22.axes.get_xaxis().set_ticks([])
ax22.axes.get_yaxis().set_ticks([])
ax23 = fig.add_subplot(gs0[5, 0])
ax23.axes.get_xaxis().set_ticks([])
ax23.axes.get_yaxis().set_ticks([])
ax24 = fig.add_subplot(gs0[6, 0])
ax24.axes.get_xaxis().set_ticks([])
ax24.axes.get_yaxis().set_ticks([])
ax25 = fig.add_subplot(gs0[7, 0])
ax25.axes.get_xaxis().set_ticks([])
ax25.axes.get_yaxis().set_ticks([])

# =============================================================================
# Box plots
# =============================================================================
k10_boxes = list(tribox_k10.values())[2]
k20_boxes = list(tribox_k20.values())[2]
k50_boxes = list(tribox_k50.values())[2]
k100_boxes = list(tribox_k100.values())[2]
k10GE_boxes = list(tribox_k10GE.values())[2]
k20GE_boxes = list(tribox_k20GE.values())[2]
k50GE_boxes = list(tribox_k50GE.values())[2]
k100GE_boxes = list(tribox_k100GE.values())[2]

'''print('k10 min/max index: %s, %s' % (np.argmin(k10_boxes), np.argmax(k10_boxes)))
print('k20 min/max index: %s, %s' % (np.argmin(k20_boxes), np.argmax(k20_boxes)))
print('k50 min/max index: %s, %s' % (np.argmin(k50_boxes), np.argmax(k50_boxes)))
print('k100 min/max index: %s, %s' % (np.argmin(k100_boxes), np.argmax(k100_boxes)))
print('')
print('k10 lowest 10:', np.argsort(k10_boxes)[:15])
print('k20 lowest 10:', np.argsort(k20_boxes)[:15])
print('k50 lowest 10:', np.argsort(k50_boxes)[:15])
print('k100 lowest 10:', np.argsort(k100_boxes)[:15])
print('')
print('k10 highest 10:', np.argsort(-1*k10_boxes)[:15])
print('k20 highest 10:', np.argsort(-1*k20_boxes)[:15])
print('k50 highest 10:', np.argsort(-1*k50_boxes)[:15])
print('k100 highest 10:', np.argsort(-1*k100_boxes)[:15])
print('')'''

#A, B, C = -1, 0, 1
#A, B, C, D = -1.5, -0.5, 0.5, 1.5
A, B, C, D, E, F, G, H = -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5
flierprops = dict(marker='>', markeredgecolor='k', markerfacecolor='k', markersize=10, linestyle='none')

k10_plot = ax1.boxplot(k10_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[H], vert=False)
for median in k10_plot['medians']:
    median.set_color('black')
k10GE_plot = ax1.boxplot(k10GE_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[G], vert=False)
for median in k10GE_plot['medians']:
    median.set_color('black')

k20_plot = ax1.boxplot(k20_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[F], vert=False)
for median in k20_plot['medians']:
    median.set_color('black')
k20GE_plot = ax1.boxplot(k20GE_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[E], vert=False)
for median in k20GE_plot['medians']:
    median.set_color('black')

k50_plot = ax1.boxplot(k50_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[D], vert=False)
for median in k50_plot['medians']:
    median.set_color('black')
k50GE_plot = ax1.boxplot(k50GE_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[C], vert=False)
for median in k50GE_plot['medians']:
    median.set_color('black')

k100_plot = ax1.boxplot(k100_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[B], vert=False)
for median in k100_plot['medians']:
    median.set_color('black')
k100GE_plot = ax1.boxplot(k100GE_boxes, sym='', widths=0.5, showfliers=False, showmeans=True, meanprops=flierprops,
                       positions=[A], vert=False)
for median in k100GE_plot['medians']:
    median.set_color('black')

s = 8 #scatter scale
a = .5 #scatter alpha
color_lime = '#999999' #gray
color_GE = '#4daf4a' #green

ax1.set_xlabel('Error', fontsize=16, labelpad=3)

singles_x1 = np.random.normal(H, 0.08, size=len(k10_boxes))
ax1.scatter(k10_boxes, singles_x1, alpha=a, s=s, c=color_lime, zorder=-10)
singles_x2 = np.random.normal(G, 0.08, size=len(k10GE_boxes))
ax1.scatter(k10GE_boxes, singles_x2, alpha=a, s=s, c=color_GE, zorder=-10)

singles_x3 = np.random.normal(F, 0.08, size=len(k20_boxes))
ax1.scatter(k20_boxes, singles_x3, alpha=a, s=s, c=color_lime, zorder=-10)
singles_x4 = np.random.normal(E, 0.08, size=len(k20GE_boxes))
ax1.scatter(k20GE_boxes, singles_x4, alpha=a, s=s, c=color_GE, zorder=-10)

singles_x5 = np.random.normal(D, 0.08, size=len(k50_boxes))
ax1.scatter(k50_boxes, singles_x5, alpha=a, s=s, c=color_lime, zorder=-10)
singles_x6 = np.random.normal(C, 0.08, size=len(k50GE_boxes))
ax1.scatter(k50GE_boxes, singles_x6, alpha=a, s=s, c=color_GE, zorder=-10)

singles_x7 = np.random.normal(B, 0.08, size=len(k100_boxes))
ax1.scatter(k100_boxes, singles_x7, alpha=a, s=s, c=color_lime, zorder=-10)
singles_x8 = np.random.normal(A, 0.08, size=len(k100GE_boxes))
ax1.scatter(k100GE_boxes, singles_x8, alpha=a, s=s, c=color_GE, zorder=-10)

if 0:
    ax1.scatter(k10_boxes[26], singles_x1[26], s=s, c='blue', zorder=10)
    ax1.scatter(k10_boxes[14], singles_x1[14], s=s, c='red', zorder=10)
    ax1.scatter(k20_boxes[26], singles_x3[26], s=s, c='blue', zorder=10)
    ax1.scatter(k20_boxes[14], singles_x3[14], s=s, c='red', zorder=10)
    ax1.scatter(k50_boxes[26], singles_x5[26], s=s, c='blue', zorder=10)
    ax1.scatter(k50_boxes[14], singles_x5[14], s=s, c='red', zorder=10)
    ax1.scatter(k100_boxes[26], singles_x7[26], s=s, c='blue', zorder=10)
    ax1.scatter(k100_boxes[14], singles_x7[14], s=s, c='red', zorder=10)

    ax1.scatter(k10GE_boxes[26], singles_x2[26], s=s, c='blue', zorder=10)
    ax1.scatter(k10GE_boxes[14], singles_x2[14], s=s, c='red', zorder=10)
    ax1.scatter(k20GE_boxes[26], singles_x4[26], s=s, c='blue', zorder=10)
    ax1.scatter(k20GE_boxes[14], singles_x4[14], s=s, c='red', zorder=10)
    ax1.scatter(k50GE_boxes[26], singles_x6[26], s=s, c='blue', zorder=10)
    ax1.scatter(k50GE_boxes[14], singles_x6[14], s=s, c='red', zorder=10)
    ax1.scatter(k100GE_boxes[26], singles_x8[26], s=s, c='blue', zorder=10)
    ax1.scatter(k100GE_boxes[14], singles_x8[14], s=s, c='red', zorder=10)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
set_box_color(k10_plot, color_lime)
set_box_color(k20_plot, color_lime)
set_box_color(k50_plot, color_lime)
set_box_color(k100_plot, color_lime)
set_box_color(k10GE_plot, color_GE)
set_box_color(k20GE_plot, color_GE)
set_box_color(k50GE_plot, color_GE)
set_box_color(k100GE_plot, color_GE)

ax1.set_xlim(0, ax1.get_xlim()[1]+2.5)
#ax1.set_yticklabels(['k10', 'k20', 'k50', 'k100'], fontsize=12)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


if 1:
    ax_tops = []
    ax_bots = []
    bar_heights = []
    ax_tops.append([item.get_ydata()[1] for item in k10_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k20_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k50_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k100_plot['caps']][1])
    ax_bots.append([item.get_ydata()[1] for item in k10_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k20_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k50_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k100_plot['caps']][0])
    ax_tops.append([item.get_ydata()[1] for item in k10GE_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k20GE_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k50GE_plot['caps']][1])
    ax_tops.append([item.get_ydata()[1] for item in k100GE_plot['caps']][1])
    ax_bots.append([item.get_ydata()[1] for item in k10GE_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k20GE_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k50GE_plot['caps']][0])
    ax_bots.append([item.get_ydata()[1] for item in k100GE_plot['caps']][0])

    # add MWU information to plot
    alt = 'less' #{'two-sided', 'less'}

    mwu_stat, pval = stats.mannwhitneyu(k10GE_boxes, k10_boxes, alternative=alt)
    print(pval)
    mwu_stat, pval = stats.mannwhitneyu(k20GE_boxes, k20_boxes, alternative=alt)
    print(pval)
    mwu_stat, pval = stats.mannwhitneyu(k50GE_boxes, k50_boxes, alternative=alt)
    print(pval)
    mwu_stat, pval = stats.mannwhitneyu(k100GE_boxes, k100_boxes, alternative=alt)
    print(pval)

    #combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    '''combinations = [(A, C), (A, B), (A, D), (B, C), (B, D), (C, D)]
    ax_bottom, ax_top = ax1.get_ylim()
    y_range = ax_top - ax_bottom
    level_idx = 3
    for x1, x2 in combinations:#range(len(comparison_methods)):
        if x1 == A and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(k10_boxes, k20_boxes, alternative=alt)
        elif x1 == B and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(k10_boxes, k50_boxes, alternative=alt)
        elif x1 == C and x2 == D:
            mwu_stat, pval = stats.mannwhitneyu(k10_boxes, k100_boxes, alternative=alt)
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
            level_idx -= 1'''
    if 0:
        ax1.set_ylim(0, ax1.get_ylim()[1]+1.5)

    #for c in list(range(-1, 2)):
        #ax1.text(c, 0 + y_range * 0.02, r'$\it{n} =$ %s' % seq_total, ha='center', size='small')

# =============================================================================
# Attribution maps for individual sequences
# =============================================================================
if 1:
    ax2.set_title('Min Error', fontsize=16, pad=10)
    start, stop = 945, 951 #rank 26

    logo = logomaker.Logo(df=wt_k10_1[start-fig_pad:stop+fig_pad+1],
                        ax=ax2,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold',
                        )

    logo = logomaker.Logo(df=wt_k10GE_1[start-fig_pad:stop+fig_pad+1],
                        ax=ax3,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold',
                        )

    logo = logomaker.Logo(df=wt_k20_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax4,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k20GE_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax5,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k50_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax6,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )
    
    logo = logomaker.Logo(df=wt_k50GE_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax7,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k100_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax8,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k100GE_1[start-fig_pad:stop+fig_pad+1],
                    ax=ax9,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )


    ax10.set_title('Max Error', fontsize=16, pad=10)
    start, stop = 770, 776 #rank 14
   
    logo = logomaker.Logo(df=wt_k10_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax10,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )
    
    logo = logomaker.Logo(df=wt_k10GE_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax11,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k20_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax12,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k20GE_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax13,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k50_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax14,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k50GE_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax15,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k100_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax16,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    logo = logomaker.Logo(df=wt_k100GE_2[start-fig_pad:stop+fig_pad+1],
                    ax=ax17,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold',
                    )

    # =============================================================================
    # Averaged attribution maps
    # =============================================================================
    ax10.set_title('Average', fontsize=16, pad=10)

    logomaker.Logo(df=avg_k10,
                    ax=ax18,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    
    logomaker.Logo(df=avg_k10GE,
                    ax=ax19,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_k20,
                    ax=ax20,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    
    logomaker.Logo(df=avg_k20GE,
                    ax=ax21,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_k50,
                    ax=ax22,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_k50GE,
                    ax=ax23,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')

    logomaker.Logo(df=avg_k100,
                    ax=ax24,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    
    logomaker.Logo(df=avg_k100GE,
                    ax=ax25,
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



