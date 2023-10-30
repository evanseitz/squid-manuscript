# =============================================================================
# Find all instances of user-defined recognition site pattern(s) across..
# ..genomic sequences. This script has two main purposes:
#
#   1 Find all instances of a given recognition site across a collection of..
#     ..genomic sequences (intended for additive or intra-motif pairwise modeling)
#   2 Additionally, if two motif inputs are provided, find all instances of..
#     ..nearest-neighbor (and second-nearest-neighbor) recognition sites..
#     (intened for additive modeling or intra and inter-motif pairwise modeling)
#
# For either category, recognition site instances are ranked using attribution..
# ..scores (using the predictions from a user-defined deep learning model) and..
# ..the number of core mutations, among other metrics. Files are saved to the..
# ..'b_recognition_sites' folder for use in subsequent scripts
# =============================================================================
# Instructions: First, make sure to adjust user inputs in 'set_parameters.py'..
#               ..as desired, following the documentation there. Only parameters..
#               ..in 'set_params_1()' are used in the current script. Before..
#               ..running this script, make sure to source the environment..
#               ..in the CLI corresponding to the chosen deep learning model
#
#               Once the environment is sourced, the current script can be run..
#               ..in the CLI via: 'python 1_locate_patterns.py'
# =============================================================================

import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings
sys.dont_write_bytecode = True
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) #turns off tensorflow warnings
logging.getLogger('tensorflow').disabled = True #turns off tensorflow warnings
import math
import pickle
import h5py
import pandas as pd
pd.set_option("display.precision", 8)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import zipfile
import shutil
import tfomics


pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
import squid.utils as squid_utils

# =============================================================================
# Import customized user parameters from script set_parameters.py
# =============================================================================
from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2

print("Importing model info, sequence data, and user parameters from set_parameters.py")

GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(pyDir, True)

num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example) #only 'pred_transform' is needed here
if pred_transform == 'pca': #override user choice since there are too few datapoints per recognition site to reliably use PCA
    pred_transform = 'sum'
save = False

if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
    userDir = os.path.join(pyDir, 'examples_CAGI5')
else:
    userDir = os.path.join(pyDir, 'examples_%s' % example)

if 'saliency' in comparison_methods:
    sys.path.append(os.path.join(userDir,'a_model_assets/scripts')) #ZULU, needs to take in function wrapper
    import saliency_embed, utils
    
# =============================================================================
# Run algorithm based on deep learning model and user-defined patterns
# =============================================================================
num_seqs = X_in.shape[0] #default is total number of sequences in test set; decrease number (int) to limit search
print('Number of sequences: %s' % num_seqs)
maxL = X_in.shape[1] #maximum length of sequence

saveDir = os.path.join(userDir, 'b_recognition_sites/%s' % (model_name))
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# generate filters
filter_A = squid_utils.seq2oh(motif_A, alphabet)
filter_A_sum = np.sum(filter_A)

if motif_B is not None:
    filter_B = squid_utils.seq2oh(motif_B, alphabet)
    filter_B_sum = np.sum(filter_B)
        
hits = np.zeros(shape=(num_seqs,maxL,2))
df_motif_A = pd.DataFrame(columns = ['seq_idx', 'motif_rank', 'motif_wt', 'motif_mutations', 'motif_start'])

# first loop: locate motif_A positions
idx = 0
for r in range(num_seqs):
    OH = X_in[r]
    X = np.expand_dims(OH, 0)
    
    if rank_type == 'saliency':
        explainer = saliency_embed.Explainer(model, example, class_index=class_idx) #tfomics.explain.Explainer(model, class_index=class_idx)
        saliency_scores = explainer.saliency_maps(X) #calculate attribution maps
        sal_scores = tfomics.explain.grad_times_input(X, saliency_scores) #reduce attribution maps to 1D scores
        x = np.expand_dims(X[0], axis=0)
        scores = np.expand_dims(saliency_scores[0], axis=0) #convert attribution maps to pandas dataframe for logomaker
        sal_df = tfomics.impress.grad_times_input_to_df(x, scores)

    for t in range(0,maxL-filter_A.shape[0]):
        hits[r,t,0] = int(np.trace(np.dot(OH[t:t+filter_A.shape[0],:], filter_A.T)))
    hits[r,:,0][hits[r,:,0] < (filter_A_sum - max_muts)] = 0

    for m in range(max_muts+1):
        hits_idx = np.argwhere(hits[r,:,0] == (filter_A_sum - m))
        for h in hits_idx:
            start_A = h[0]
            stop_A = start_A+len(motif_A)
            df_motif_A.at[idx, 'seq_idx'] = r
            if rank_type == 'saliency':
                # calculate saliency score for motif A and motif B cores
                sal_df_A = np.array(sal_df)[start_A:stop_A,:]
                score_A = np.trace(np.dot(sal_df_A, filter_A.T))
            elif rank_type == 'ISM':

                ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                        unwrap_prediction, compress_prediction, pred_transform, 
                                                        pred_trans_delimit, log2FC, max_in_mem, save, saveDir,
                                                        start=start_A, stop=stop_A)
                ISM_logo_A = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
                #score_A = np.trace(np.dot(np.array(ISM_logo_A[start_A:stop_A]), filter_A.T))
                score_A = np.sum(np.array(ISM_logo_A[start_A:stop_A]))                        
                
                if 0: #optional sanity check to view each discovered recognition site instance
                    import logomaker
                    print(ISM_logo_A[start_A:stop_A])
                    fig, ax = plt.subplots()
                    logomaker.Logo(df=ISM_logo_A[start_A-3:stop_A+3],
                                    ax=ax,
                                    fade_below=.5,
                                    shade_below=.5,
                                    width=.9,
                                    center_values=False,
                                    color_scheme='classic',
                                    font_name='Arial Rounded MT Bold')
                    ax.set_title('muts=%s, score=%s' % (m, score_A))
                    plt.show()
                
            df_motif_A.at[idx, 'motif_rank'] = score_A
            df_motif_A.at[idx, 'motif_mutations'] = m
            df_motif_A.at[idx, 'motif_start'] = start_A
            # transcribe WT sequence for motif
            oh = OH[start_A:stop_A]
            seq = squid_utils.oh2seq(oh, alphabet)
            df_motif_A.at[idx, 'motif_wt'] = seq
            idx += 1

    if motif_B is not None:
        for t in range(0,maxL-filter_B.shape[0]):
            hits[r,t,1] = int(np.trace(np.dot(OH[t:t+filter_B.shape[0],:], filter_B.T)))
        hits[r,:,1][hits[r,:,1] < (filter_B_sum - max_muts)] = 0

df_motif_A = df_motif_A.sort_values(by=['motif_rank'], ascending=False)

# save dataframe to file:
df_motif_A.to_csv(os.path.join(saveDir, '%s_positions.csv' % motif_A_name), index=False)
print('%s_positions saved to file.' % motif_A_name)

# second loop: find and rank nearest-neighbor motif instances
if motif_B is not None:
    hits_both = np.array(hits[:,:,0], copy=True)
    hits_both[:,:][hits_both[:,:] > 0] = 1
    for r in range(num_seqs):
        hits_both[r,:] += hits[r,:,1]

    seq_idx = []
    motif_rank = []
    motif_mutations = []
    motif_start = []
    inter_dist = []
    motif_wt = []

    for r in range(num_seqs):
        total = len(hits_both[r,:][hits_both[r,:] > 0])
        if total > 1:
            OH = X_in[r]
            X = np.expand_dims(OH, 0)
            pos = np.argwhere(hits_both[r,:] > 0) #locate motif hits
            
            if rank_type == 'saliency':
                explainer = saliency_embed.Explainer(model, example, class_index=class_idx) #tfomics.explain.Explainer(model, class_index=class_idx)
                saliency_scores = explainer.saliency_maps(X) #calculate attribution maps
                sal_scores = tfomics.explain.grad_times_input(X, saliency_scores) #reduce attribution maps to 1D scores
                x = np.expand_dims(X[0], axis=0)
                scores = np.expand_dims(saliency_scores[0], axis=0) #convert attribution maps to pandas dataframe for logomaker
                sal_df = tfomics.impress.grad_times_input_to_df(x, scores)

            # define nearest-neighbor instances
            for d in range(total-1):
                if hits_both[r,int(pos[d])] == 1:
                    motif_LHS = motif_A
                    filter_LHS = filter_A
                elif hits_both[r,int(pos[d])] > 1:
                    motif_LHS = motif_B
                    filter_LHS = filter_B
                if hits_both[r,int(pos[(d+1)])] == 1:
                    motif_RHS = motif_A
                    filter_RHS = filter_A
                elif hits_both[r,int(pos[(d+1)])] > 1:
                    motif_RHS = motif_B
                    filter_RHS = filter_B

                motif_dist = int(pos[d+1]) - (int(pos[d]) + len(motif_LHS)) #central distance
                if motif_dist >= 0 and motif_dist <= max_dist: #(ignore overlapping instances) and (set upper-distance threshold)
                    inter_dist.append(motif_dist)
                    motif_start.append(int(pos[d]))
                    seq_idx.append(r)
                    # calculate combined number of core mutations in both motifs
                    start_A = int(pos[d])
                    stop_A = int(pos[int(d)]+len(motif_LHS))
                    start_B = int(pos[int(d+1)])
                    stop_B = int(pos[int(d+1)]+len(motif_RHS))
                    convol_A = int(np.trace(np.dot(OH[start_A:stop_A,:], filter_LHS.T)))
                    convol_B = int(np.trace(np.dot(OH[start_B:stop_B,:], filter_RHS.T)))
                    motif_mutations.append(int((filter_A_sum + filter_B_sum) - (convol_A + convol_B)))
                    # transcribe wt sequence
                    oh = OH[start_A:stop_B]
                    motif_wt.append(squid_utils.oh2seq(oh, alphabet))
                    #calculate attribution score for motif A and motif B cores
                    if rank_type == 'saliency':
                        sal_A = np.array(sal_df)[start_A:stop_A,:]
                        score_A = np.trace(np.dot(sal_A, filter_LHS.T))
                        sal_B = np.array(sal_df)[start_B:stop_B,:]
                        score_B = np.trace(np.dot(sal_B, filter_RHS.T))
                    elif rank_type == 'ISM':
                        ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                                unwrap_prediction, compress_prediction, pred_transform, 
                                                                pred_trans_delimit, log2FC, max_in_mem, save, saveDir,
                                                                start=start_A, stop=stop_A)
                        ISM_logo_A = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
                        #score_A = np.trace(np.dot(ISM_logo_A[start_A:stop_A], filter_LHS.T))
                        score_A = np.sum(np.array(ISM_logo_A[start_A:stop_A]))
                        
                        ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction, 
                                                                unwrap_prediction, compress_prediction, pred_transform, 
                                                                pred_trans_delimit, log2FC, max_in_mem, save, saveDir,
                                                                start=start_B, stop=stop_B)
                        ISM_logo_B = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
                        #score_B = np.trace(np.dot(ISM_logo_B[start_B:stop_B], filter_RHS.T))
                        score_B = np.sum(np.array(ISM_logo_B[start_B:stop_B]))
                    motif_rank.append(score_A + score_B)

            # define 2nd-neighbor instances (leap frog)
            for d in range(total-2):
                if hits_both[r,int(pos[d])] == 1:
                    motif_LHS = motif_A
                    filter_LHS = filter_A
                elif hits_both[r,int(pos[d])] > 1:
                    motif_LHS = motif_B
                    filter_LHS = filter_B
                if hits_both[r,int(pos[(d+2)])] == 1:
                    motif_RHS = motif_A
                    filter_RHS = filter_A
                elif hits_both[r,int(pos[(d+2)])] > 1:
                    motif_RHS = motif_B
                    filter_RHS = filter_B
            
                motif_dist = int(pos[d+2]) - (int(pos[d]) + len(motif_LHS)) #central distance
                if motif_dist >= 0 and motif_dist <= max_dist: #(ignore overlapping instances) and (set upper-distance threshold)
                    inter_dist.append(motif_dist)
                    motif_start.append(int(pos[d]))
                    seq_idx.append(r)
                    # calculate combined number of core mutations in both motifs
                    start_A = int(pos[d])
                    stop_A = int(pos[int(d)]+len(motif_LHS))
                    start_B = int(pos[int(d+2)])
                    stop_B = int(pos[int(d+2)]+len(motif_RHS))
                    convol_A = int(np.trace(np.dot(OH[start_A:stop_A,:], filter_LHS.T)))
                    convol_B = int(np.trace(np.dot(OH[start_B:stop_B,:], filter_RHS.T)))
                    motif_mutations.append(int((filter_A_sum + filter_B_sum) - (convol_A + convol_B)))
                    # transcribe wt sequence
                    oh = OH[start_A:stop_B]
                    motif_wt.append(squid_utils.oh2seq(oh, alphabet))
                    #calculate attribution score for motif A and motif B cores
                    if rank_type == 'saliency':
                        sal_A = np.array(sal_df)[start_A:stop_A,:]
                        score_A = np.trace(np.dot(sal_A, filter_LHS.T))
                        sal_B = np.array(sal_df)[start_B:stop_B,:]
                        score_B = np.trace(np.dot(sal_B, filter_RHS.T))
                    elif rank_type == 'ISM':
                        ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                                unwrap_prediction, compress_prediction, pred_transform, 
                                                                pred_trans_delimit, log2FC, max_in_mem, save, saveDir,
                                                                start=start_A, stop=stop_A)
                        ISM_logo_A = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
                        #score_A = np.trace(np.dot(ISM_logo_A[start_A:stop_A], filter_LHS.T))
                        score_A = np.sum(np.array(ISM_logo_A[start_A:stop_A]))
                        
                        ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                                unwrap_prediction, compress_prediction, pred_transform, 
                                                                pred_trans_delimit, log2FC, max_in_mem, save, saveDir,
                                                                start=start_B, stop=stop_B)
                        ISM_logo_B = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
                        #score_B = np.trace(np.dot(ISM_logo_B[start_B:stop_B], filter_RHS.T))
                        score_B = np.sum(np.array(ISM_logo_B[start_B:stop_B]))
                    motif_rank.append(score_A + score_B)
                    

    df_dists = pd.DataFrame(list(zip(seq_idx, motif_rank, motif_wt, motif_mutations, motif_start, inter_dist)), columns=['seq_idx','motif_rank','motif_wt','motif_mutations','motif_start','inter_dist'])
    df_dists = df_dists.sort_values(by=['motif_rank'], ascending=False)

    # save dataframe to file:
    df_dists.to_csv(os.path.join(saveDir, '%s_%s_positions.csv' % (motif_A_name, motif_B_name)), index=False)
    print('%s_%s_positions saved to file.' % (motif_A_name, motif_B_name))
    
    
    
# save histogram(s) of motif rank frequencies
for total_muts in range(max_muts+1):
    motif_info = df_motif_A
    motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
    motif_info = motif_info.loc[motif_info['motif_mutations'] == total_muts]
    motif_info.reset_index(drop=True,inplace=True)
    
    plt.title('%s recognition sites | %s core mutations' % (motif_A_name, total_muts))
    plt.hist(motif_info['motif_rank'], bins=int(np.sqrt(len(motif_info['motif_rank']))))
    plt.ylabel('Frequency')
    plt.xlabel('Motif rank (%s)' % rank_type)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, '%s_ranks_mut%s.png' % (motif_A_name, total_muts)), facecolor='w', dpi=200)
    plt.close()
    
if motif_B is not None:
    for total_muts in range(max_muts+1):
        motif_info = df_dists
        motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
        motif_info = motif_info.loc[motif_info['motif_mutations'] == total_muts]
        motif_info.reset_index(drop=True,inplace=True)
        
        plt.title('%s_%s recognition sites | %s core mutations' % (motif_A_name, motif_B_name, total_muts))
        plt.hist(motif_info['motif_rank'], bins=int(np.sqrt(len(motif_info['motif_rank']))))
        plt.ylabel('Frequency')
        plt.xlabel('Motif rank (%s)' % rank_type)
        plt.tight_layout()
        plt.savefig(os.path.join(saveDir, '%s_%s_ranks_mut%s.png' % (motif_A_name, motif_B_name, total_muts)), facecolor='w', dpi=200)
        plt.close()
    
    