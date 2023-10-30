# =============================================================================
# TBD
# =============================================================================
# Instructions: Before running, make sure to source the correct environment in..
#               ..the CLI. Next, customize the variables in 'set_paramaters.py'..
#               ..to match the settings used for running '1_locate_patterns.py'
#               The current script can be run via: python conditional_pruning.py
# =============================================================================


import os, sys
import pandas as pd
import numpy as np
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

if 'saliency' in comparison_methods:
    import tfomics
    sys.path.append(os.path.join(userDir,'a_model_assets/scripts'))
    import saliency_embed, utils

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
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 400].index) #400
        motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 600].index) #600
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

else:
    filter_by_mut = False
        

df_pruned = pd.DataFrame(columns = ['seq_idx', 'motif_rank', 'motif_wt', 'motif_mutations', 'motif_start', 'y_ratio'])
max_prune = 100 #total number of pruned sequences desired
ratio_threshold = 0.1#0.6 #minimum cutoff for ratio of max(site_y) to max(sequence_y); see below

# prune motif A dataframe given desired conditions:
if motif_A_name is not None:
    print('%s pruning progress:' % (motif_A_name))
    idx = 0
    for i in range(len(motif_info)):
        start = motif_info.at[i,'motif_start']
        if 1: #standard pruning for high-activity sites
            # compare local activity at motif site to max activity of entire sequence
            if 'saliency' in comparison_methods:
                explainer = saliency_embed.Explainer(model, class_index=class_idx) #tfomics.explain.Explainer(model, class_index=class_idx)
                saliency_scores = explainer.saliency_maps(np.expand_dims(X_in[i], 0))
                scores = np.expand_dims(saliency_scores[0], axis=0)
                logo_full = tfomics.impress.grad_times_input_to_df(np.expand_dims(X_in[i], 0), scores)
                
            else:
                df_full = squid_utils.ISM_single(X_in[i], model, class_idx, example, get_prediction,
                                                    unwrap_prediction, compress_prediction, pred_transform, None)
                logo_full = squid_utils.l2_norm_to_df(X_in[i], df_full, alphabet=alphabet, alpha=alpha)
            
            score_site = np.max(np.array(logo_full[start:start+len(motif_A)]))
            score_full = np.amax(np.array(logo_full))
            score_ratio = score_site / score_full
            if score_ratio >= ratio_threshold:
                df_pruned.at[i, 'seq_idx'] = motif_info['seq_idx'][i]
                df_pruned.at[i, 'motif_rank'] = motif_info['motif_rank'][i]
                df_pruned.at[i, 'motif_wt'] = motif_info['motif_wt'][i]
                df_pruned.at[i, 'motif_mutations'] = motif_info['motif_mutations'][i]
                df_pruned.at[i, 'motif_start'] = motif_info['motif_start'][i]
                df_pruned.at[i, 'y_ratio'] = score_ratio
                print('%s / %s complete' % (idx+1, max_prune))
                if idx == max_prune-1:
                    break
                idx += 1   


    if filter_by_mut is True:
        df_pruned.to_csv(os.path.join(userDir, '%s_positions_pruned_mut%s.csv' % (motif_A_name, use_mut)), index=False)
        print('%s_positions pruned with condition saved to file as %s_positions_pruned_mut%s.' % (motif_A_name, motif_A_name, use_mut))
    else:
        
        df_pruned = df_pruned.sort_values(by = ['seq_idx','y_ratio'], ascending = [True, False])
        df_pruned.to_csv(os.path.join(userDir, '%s_positions_pruned.csv' % motif_A_name), index=False)
        print('%s_positions pruned with condition saved to file as %s_positions_pruned.' % (motif_A_name, motif_A_name))



