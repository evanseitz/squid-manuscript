# =============================================================================
# TBD
# =============================================================================
# Instructions: Before running, make sure to source the correct environment in..
#               ..the CLI. Next, customize the variables in 'set_paramaters.py'..
#               ..to match the settings used for running '1_locate_patterns.py'
#               The current script can be run via: python conditional_pruning.py
# =============================================================================


import os, sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings

pyDir = os.path.dirname(os.path.abspath(__file__))
parDir = os.path.dirname(pyDir)
grandParDir = os.path.dirname(parDir)
sys.path.append(pyDir)
sys.path.append(parDir)
sys.path.append(grandParDir)

import squid.utils as squid_utils
from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2


# =============================================================================
# Load data
# =============================================================================
print("Importing model info, sequence data, and user parameters from set_parameters.py")

GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_mut, max_dist, rank_type,\
comparison_methods, model_name, class_idx, alphabet, alpha, bin_res, model, X_in = set_params_1(parDir, True)
    
num_sim, pred_transform, scope, sort, use_mut, model_pad, compare, map_crop, clear_RAM, save = set_params_2(example)

userDir = os.path.join(parDir, 'examples_%s/b_recognition_sites/%s' % (example, model_name))

if scope == 'intra':
    motif_info = pd.read_csv(os.path.join(userDir,'%s_positions.csv' % (motif_A_name)))
elif scope == 'inter':
    motif_info = pd.read_csv(os.path.join(userDir,'%s_%s_distances.csv' % (motif_A_name, motif_B_name)))


# =============================================================================
# Recreate dataframe(s) without duplicate entries
# =============================================================================
if sort is True:
    if scope == 'intra':
        motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
    elif scope == 'inter':
        motif_info = motif_info.sort_values(by = ['motif_rank','inter_dist'], ascending = [False, True])
        
    if example == 'BPNet':
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 400].index)
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 600].index)
        motif_info.reset_index(drop=True,inplace=True)
    if example == 'DeepSTARR':
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 80].index)
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 170].index)
        motif_info.reset_index(drop=True,inplace=True)
    
    filter_by_mut = True #only prune dataframe for sites having 'use_mut' core mutations
    if filter_by_mut is True:
        motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]
        
    motif_info.reset_index(drop=True,inplace=True)
    # sort one-hot encodings based on new ordering (needed for plotting WT information in one figure below)
    motif_info_idx = motif_info['seq_idx']
    X_in = X_in[motif_info_idx]
    
motif_info_top = motif_info[0:50]
print(motif_info_top['motif_wt'].value_counts())

counts = np.array(list(motif_info_top['motif_wt'].value_counts()))
#print(counts)

from scipy.stats import entropy
E = entropy(counts/len(counts), base=2)
print('entropy:',E)

fig, ax = plt.subplots(figsize=[5., 2])#nrows=1, ncols=1, figsize=[5,1.5])

#plt.bar(counts)
ax = motif_info_top['motif_wt'].value_counts().plot(kind='bar', fontsize=6) #barh
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel('Count', fontsize=10, labelpad=10)
plt.text(.99, .965, r'$S = %0.2f$' % E, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)

plt.tight_layout()
plt.show()
