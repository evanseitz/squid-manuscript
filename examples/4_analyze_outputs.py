# =============================================================================
# Compare and analyze statistical and biophysical properties of an ensemble..
# of attribution maps produced by surrogate models 
#
# In the previous script '3_surrogate_modeling.py', a collection of surrogate..
# ..models were produced–optionally alongside gold-standard methods–forming..
# ..attribution maps, each corresponding to a specific genomic locus housing..
# ..an instance of the same putative recognition site of interest. The current..
# ..script provides algorithms to analyze and compare the bulk properties of..
# ..these ensembles. Available outputs include:
#   1. Standardization options for providing a direct comparison between..
#       ..surrogate models and gold-standard attribution maps
#   2. Averaged surrogate models and averaged gold-standard attribution maps
#   3. Boxplots for analyzing robustness of each model/attribution map
#   4. Plots of positional standard deviation across each ensemble and..
#       ..success rate, as compared to the consensus motif
#   5. Page-length plots comparing each attribution map per sequence instance..
#       ..color coded to represent deviances from the consensus motif
#   6. Methods to crop each inter-motif pairwise model (initially representing..
#       ..motif instances spaced some arbitrary distance apart) and overlay..
#       ..them on the same plot to form averaged pairwise plots and statistics
# =============================================================================
# Instructions: First, make sure that user inputs in 'set_parameters.py' are..
#               ..identical to those used in previous scripts during the current..
#               ..analysis. Additional parameters pertaining to the current..
#               ...analysis should be customized in 'set_params_4()'. Also make..
#               ..sure to source the correct environment in the CLI corresponding..
#               ..to the chosen surrogate model used for MAVE-NN analysis. Finally..
#               ..run the current script in the CLI via 'python 4_analyze_outputs.py'
# =============================================================================

import os, sys
sys.dont_write_bytecode = True

def op(pyDir, flank_idx, max_flank, gauge_type):

    import time
    import shutil
    import numpy as np
    from numpy import linalg as LA
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from itertools import groupby
    import scipy
    from scipy.spatial.distance import hamming 
    import pickle
    import mavenn
    import logomaker
    import h5py
    import zipfile

    pyDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(pyDir)
    sys.path.append(pyDir)
    sys.path.append(parentDir)
    import squid.utils as squid_utils
    import squid.figs_surrogate as squid_figs


    # =============================================================================
    # Import customized user parameters from script set_parameters.py
    # =============================================================================
    from set_parameters import set_params_1, set_params_2, set_params_3, set_params_4
    
    if max_flank is None: #don't show if using batch loop
        print("Importing model, sequence data, and user parameters from set_parameters.py")

    GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
    comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(pyDir, False)
        
    if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
        userDir = os.path.join(pyDir, 'examples_CAGI5')
    else:
        userDir = os.path.join(pyDir, 'examples_%s' % example)

    num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)
        
    surrogate, regression, gpmap, gauge_mavenn, linearity, noise, noise_order, drop = set_params_3()

    seq_total, gauge, show_cropped, show_compared, fig_pad, filter_muts, standardize_local, standardize_global = set_params_4()

    if max_flank is not None:
        #print('max_flank for loop:', max_flank)
        print('gauge_type for loop:', gauge_type)
        gauge = gauge_type
    
    fig_pad = flank_idx #overwrite to make program loopable

    if int(model_pad) < int(fig_pad): #'model_pad' must be same as used to generate data
        print("Error: parameter 'fig_pad' cannot exceed 'model_pad'.")

    if 1: #default analysis
        mask = False
        mask_pad = None
    else: #mask out central motif to examine attribution errors only over background 
        mask = True
        mask_pad = 3

    # =============================================================================
    # Rerun algorithms needed to match setup used in previous script
    # =============================================================================
    if scope == 'intra':
        dataDir = os.path.join(userDir, 'c_surrogate_outputs/%s/SQUID_%s_%s_mut%s' % (model_name, motif_A_name, scope, use_mut))
        saveDir = os.path.join(userDir, 'd_outputs_analysis/%s/SQUID_%s_%s_mut%s/pad%s' % (model_name, motif_A_name, scope, use_mut, fig_pad))
        motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s/%s_positions.csv' % (model_name, motif_A_name)))
    elif scope == 'inter':
        dataDir = os.path.join(userDir, 'c_surrogate_outputs/%s/SQUID_%s_%s_%s_mut%s' % (model_name, motif_A_name, motif_B_name, scope, use_mut))
        saveDir = os.path.join(userDir, 'd_outputs_analysis/%s/SQUID_%s_%s_%s_mut%s/pad%s' % (model_name, motif_A_name, motif_B_name, scope, use_mut, fig_pad))
        motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s/%s_%s_positions.csv' % (model_name, motif_A_name, motif_B_name)))

    #print(motif_info.head(50))

    total_seq_num = len(next(os.walk(dataDir))[1])
    if seq_total > total_seq_num:
        seq_total = total_seq_num

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # =============================================================================
    # Load and sort motif-location dataframe by a given condition (must match upstream scripts)
    # =============================================================================
    if sort is True:
        if scope == 'intra':
            motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
        elif scope == 'inter':
            motif_info = motif_info.sort_values(by = ['motif_rank','inter_dist'], ascending = [False, True])
        motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]
        
        if example == 'BPNet':
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 400].index)
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 600].index)
        if example == 'DeepSTARR':
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 80].index) #110 for gata, 80 for creb
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 170].index) #140 for gata, 170 for creb
            
    else:
        motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]

    motif_info.reset_index(drop=True, inplace=True)
    # sort one-hot encodings based on new ordering (needed for plotting WT information in one figure below)
    motif_info_idx = motif_info['seq_idx']
    X_in = X_in[motif_info_idx]
        
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    if max_flank == None:
        print(motif_info)
        
    WT_seqs_A = [] #store wild-type nucleotides for each sequence
    maxL = X_in.shape[1] #maximum length of sequence
    motif_A_len = len(motif_A)
    if scope == 'intra':
        motif_idxs = [0]
    elif scope == 'inter':
        motif_idxs = [0,1]
        motif_B_len = len(motif_B)
        WT_seqs_B = []

        import re
        dist_x = []
        dist_y = []
        def get_trailing_number(s):
            m = re.search(r'\d+$', s)
            return int(m.group()) if m else None
        

    # global matrices (for eventual averaging)
    add_A_all = np.zeros((seq_total,motif_A_len+2*fig_pad,4))
    add_A_all_norm = np.zeros((seq_total,motif_A_len+2*fig_pad,4))
    pw_A_all = np.zeros((seq_total,motif_A_len+2*fig_pad,4,motif_A_len+2*fig_pad,4))
    if scope == 'inter':
        add_B_all = np.zeros((seq_total,motif_B_len+2*fig_pad,4))
        add_B_all_norm = np.zeros((seq_total,motif_B_len+2*fig_pad,4))
        pw_B_all = np.zeros((seq_total,motif_B_len+2*fig_pad,4,motif_B_len+2*fig_pad,4))
        pw_I_all = np.zeros((seq_total,motif_A_len+motif_B_len+4*fig_pad,4,motif_A_len+motif_B_len+4*fig_pad,4))
    if compare is True:
        ISM_A_all = np.zeros((seq_total,motif_A_len+2*fig_pad,4))
        ISM_A_all_norm = np.zeros((seq_total,motif_A_len+2*fig_pad,4))
        sal_A_all = np.zeros((seq_total,motif_A_len+2*fig_pad,4)) #corresponds to 'other' attr map if not saliency map
        sal_A_all_norm = np.zeros((seq_total,motif_A_len+2*fig_pad,4)) #corresponds to 'other' attr map if not saliency map
        if scope == 'inter':
            ISM_B_all = np.zeros((seq_total,motif_B_len+2*fig_pad,4))
            ISM_B_all_norm = np.zeros((seq_total,motif_B_len+2*fig_pad,4))
            sal_B_all = np.zeros((seq_total,motif_B_len+2*fig_pad,4)) #corresponds to 'other' attr map if not saliency map
            sal_B_all_norm = np.zeros((seq_total,motif_B_len+2*fig_pad,4)) #corresponds to 'other' attr map if not saliency map

    if filter_muts is True:
        filter_muts_A = []
        filter_muts_B = []
                
    # =============================================================================
    # Load in final model parameters per sequence
    # =============================================================================
    for df_idx in range(0,seq_total):
        if max_flank == None:
            print('Sequence index:',df_idx)
        for folder in os.listdir(dataDir):
            if folder.startswith('rank%s_' % (df_idx)):
                path = os.path.join(dataDir, folder)
                            
                if scope == 'intra':
                    motif_wt = motif_info.loc[df_idx][2]
                    startA = motif_info.loc[df_idx][4]
                    stopA = startA + motif_A_len
                    if model_pad != 'full':
                        start_full = startA - model_pad
                        stop_full = stopA + model_pad
                    else:
                        start_full = 0
                        stop_full = maxL
                    motif_wt_A = motif_wt[0:len(motif_A)]
                    
                elif scope == 'inter':
                    motif_wt = motif_info.loc[df_idx][2]
                    startA = motif_info.loc[df_idx][4]
                    stopA = startA + motif_A_len
                    dist = motif_info.loc[df_idx][5]
                    startB = stopA + dist
                    stopB = startB + motif_B_len
                    if model_pad != 'full':
                        start_full = startA - model_pad
                        stop_full = stopB + model_pad
                    else:
                        start_full = 0
                        stop_full = maxL
                    motif_wt_A = motif_wt[0:len(motif_A)]
                    motif_wt_B = motif_wt[len(motif_A)+dist:len(motif_A)+dist+len(motif_B)]
                    # get positions for extracting from index-zero matrices
                    pos_1 = startA - start_full - fig_pad
                    pos_2 = pos_1 + motif_A_len + 2*fig_pad
                    pos_3 = pos_2 + dist - 2*fig_pad
                    pos_4 = pos_3 + motif_B_len + 2*fig_pad

                    if filter_muts is True:
                        Ns_in_A = len([i for i in motif_A if i=='N'])
                        hamming_A = int((hamming(np.array(list(motif_A)), np.array(list(motif_wt_A))))*len(motif_A))
                        if hamming_A > Ns_in_A: #presence of at least one single, non-trivial mutation (alter this definition as required)
                            filter_muts_A.append(df_idx)

                        if scope == 'inter':
                            Ns_in_B = len([i for i in motif_B if i=='N'])
                            hamming_B = int((hamming(np.array(list(motif_B)), np.array(list(motif_wt_B))))*len(motif_B))
                            if hamming_B > Ns_in_B: #presence of at least one single, non-trivial mutation (alter this definition as required)
                                filter_muts_B.append(df_idx)

                    dist_x.append(int(get_trailing_number(folder)))
                    

                if surrogate == 'mavenn':
                    if linearity == 'linear':
                        mavenn_additive = pd.read_csv(os.path.join(path, 'logo_additive_linear.csv'), index_col=0)
                    elif linearity == 'nonlinear':
                        try:
                            mavenn_additive = pd.read_csv(os.path.join(path, 'logo_additive.csv'), index_col=0)
                        except OSError:
                            mavenn_additive = pd.read_csv(os.path.join(path, 'mavenn_additive.csv'), index_col=0)

                elif surrogate == 'ridge':
                    mavenn_additive = pd.read_csv(os.path.join(path, 'ridge_additive.csv'), index_col=0)


                # fix gauge for fair comparison to other attribution maps
                #if gauge != 'empirical': #else skip gauge transform since MAVE-NN default is 'empirical'
                mavenn_additive = squid_utils.fix_gauge(np.array(mavenn_additive), gauge=gauge, wt=X_in[df_idx])

                if gpmap == 'pairwise':
                    mavenn_pairwise = np.load(os.path.join(path,'mavenn_pairwise.npy'))
                    mavenn_pairwise_copy = np.array(mavenn_pairwise, copy=True)

                if compare is True: #load in gold-standard attribution maps
                    if 'ISM_single' in comparison_methods:
                        ISM_single = np.load(os.path.join(path,'attributions_ISM_single.npy'))
                        # fix gauge for fair comparison to other attribution maps
                        ISM_single = squid_utils.fix_gauge(np.array(ISM_single), gauge=gauge, wt=X_in[df_idx])

                    if 'saliency' in comparison_methods:
                        other = np.load(os.path.join(path,'attributions_saliency.npy'))

                    elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                        other = pd.read_csv(os.path.join(path,'attributions_deepLIFT_hypothetical.csv'), index_col=0)
                        if 0:
                            other = pd.read_csv(os.path.join(path,'attributions_deepLIFT_contribution.csv'), index_col=0)
                        
                    if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                        # fix gauge for fair comparison to other attribution maps
                        other = squid_utils.fix_gauge(np.array(other), gauge=gauge, wt=X_in[df_idx])

                    if gpmap == 'pairwise':
                        ISM_double = np.load(os.path.join(path,'attributions_ISM_double.npy'))

                if mask is True:
                    mavenn_additive[startA-mask_pad:stopA+mask_pad] = 0.
                    ISM_single[startA-mask_pad:stopA+mask_pad] = 0.
                    other[startA-mask_pad:stopA+mask_pad] = 0.
                            
                # isolate motif elements
                WT_seqs_A.append(squid_utils.oh2seq(X_in[df_idx],alphabet)[startA-fig_pad:stopA+fig_pad])
                add_A = np.array(mavenn_additive)[startA-fig_pad:stopA+fig_pad]
                if scope == 'inter':
                    WT_seqs_B.append(squid_utils.oh2seq(X_in[df_idx],alphabet)[startB-fig_pad:stopB+fig_pad])
                    add_B = np.array(mavenn_additive)[startB-fig_pad:stopB+fig_pad]
                                            
                add_full = np.array(mavenn_additive)[start_full:stop_full]

                if compare is True:
                    if 'ISM_single' in comparison_methods:
                        ISM_A = np.array(ISM_single)[startA-fig_pad:stopA+fig_pad]
                        ISM_full = np.array(ISM_single)[start_full:stop_full]
                    if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                        sal_A = np.array(other)[startA-fig_pad:stopA+fig_pad]
                        sal_full = np.array(other)[start_full:stop_full]
                    if scope == 'inter':
                        if 'ISM_single' in comparison_methods:
                            ISM_B = np.array(ISM_single)[startB-fig_pad:stopB+fig_pad]
                        if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                            sal_B = np.array(other)[startB-fig_pad:stopB+fig_pad]
                            

                if pred_transform == 'pca': # ad-hoc sense correction
                    if 0:
                        """
                        For use if a mishap occured with standard sense correction due to unexpected behavior
                        When this option is enabled for a given rank_idx corresponding to a problematic..
                        ..sequence, a new mavenn_additive.csv will be saved in the current outputs folder..
                        ..with the sense recorrected. Replace the original file with this sense-corrected..
                        ..version and finally make sure to disable this section during subsequent runs
                        """
                        if df_idx in [49]: #e.g., for sequence with folder titled rank44_seq17519
                            if 0: #reverse sense of additive
                                add_A *= -1.
                                add_full *= -1.
                                mavenn_temp = np.array(mavenn_additive) * -1.
                                mavenn_temp = squid_utils.arr2pd(mavenn_temp, alphabet)
                                #mavenn_temp.to_csv(os.path.join(saveDir, 'mavenn_additive_%s.csv' % df_idx))
                                mavenn_temp.to_csv(os.path.join(path, 'mavenn_additive.csv')) #directly replace old files

                            if 1: #reverse sense of ISM
                                ISM_A *= -1
                                ISM_single *= -1.
                                ISM_full *= -1.
                                #np.save(os.path.join(saveDir, 'attributions_ISM_single_%s.npy' % df_idx), ISM_single)
                                np.save(os.path.join(path, 'attributions_ISM_single.npy'), ISM_single) #directly replace old files
                    

                # update global motif matrices (for eventual averaging)
                if standardize_local is True: #standardize arrays
                    add_A_all_norm[df_idx,:,:] = squid_utils.normalize(add_A, add_full)
                    if scope == 'inter':
                        add_B_all_norm[df_idx,:,:] = squid_utils.normalize(add_B, add_full)
                    if compare is True:
                        if 'ISM_single' in comparison_methods:
                            ISM_A_all_norm[df_idx,:,:] = squid_utils.normalize(ISM_A, ISM_full)
                        if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                            sal_A_all_norm[df_idx,:,:] = squid_utils.normalize(sal_A, sal_full)
                        if scope == 'inter':
                            if 'ISM_single' in comparison_methods:
                                ISM_B_all_norm[df_idx,:,:] = squid_utils.normalize(ISM_B, ISM_full)
                            if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                                sal_B_all_norm[df_idx,:,:] = squid_utils.normalize(sal_B, sal_full)

                add_A_all[df_idx,:,:] = add_A
                if scope == 'inter':
                    add_B_all[df_idx,:,:] = add_B
                if compare is True:
                    if 'ISM_single' in comparison_methods:
                        ISM_A_all[df_idx,:,:] = ISM_A
                    if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                        sal_A_all[df_idx,:,:] = sal_A
                    if scope == 'inter':
                        if 'ISM_single' in comparison_methods:
                            ISM_B_all[df_idx,:,:] = ISM_B
                        if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                            sal_B_all[df_idx,:,:] = sal_B
                        
                # update start/end positions of motifs in pairwise matrix
                if gpmap == 'pairwise':
                    # isolate pairwise elements
                    pw_A = mavenn_pairwise[pos_1:pos_2,:,pos_1:pos_2,:] #motif A PW segment
                    pw_B = mavenn_pairwise[pos_3:pos_4,:,pos_3:pos_4,:] #motif B PW segment
                    pw_I = mavenn_pairwise_copy[pos_1:pos_4,:,pos_1:pos_4,:] #A/B interaction PW segment
                    pw_I[0:motif_A_len+dist,:,0:motif_A_len+dist,:] = 0
                    pw_I[motif_A_len+2*fig_pad:,:,motif_A_len+2*fig_pad:,:] = 0
                    pw_I = pw_I[0:motif_A_len+motif_B_len+4*fig_pad, :, -(motif_A_len+motif_B_len+4*fig_pad):motif_A_len+dist+motif_B_len+4*fig_pad, :]

                    #dist_y.append(np.sum(pw_I))
                    dist_y.append(np.linalg.norm(pw_I))#, axis=(1,2)))

                    # update global pairwise matrices (for eventual averaging)
                    pw_A_all[df_idx,:,:,:,:] = pw_A
                    pw_B_all[df_idx,:,:,:,:] = pw_B
                    pw_I_all[df_idx,:,:,:,:] = pw_I
                
                xticks_A = np.arange(startA-fig_pad, stopA+fig_pad, 1) #update x-axis tick labels for plots
                if scope == 'inter':
                    xticks_B = np.arange(startB-fig_pad, stopB+fig_pad, 1)
                    xticks_I = np.concatenate([xticks_A, xticks_B])
                                
                # =============================================================================
                # For each sequence, plot cropped logos/matrices from additive/pairwise models
                # =============================================================================
                if show_cropped is True:
                    # =============================================================================
                    # Plot additive A/B terms per sequence             
                    # =============================================================================
                    # evaluate min/max values for colormap
                    if np.abs(np.amin(np.array(mavenn_additive[start_full:stop_full]))) < np.abs(np.amax(np.array(mavenn_additive[start_full:stop_full]))):
                        clim = np.abs(np.amax(np.array(mavenn_additive[start_full:stop_full])))
                    else:
                        clim = np.abs(np.amin(np.array(mavenn_additive[start_full:stop_full])))
                    for motif_idx in motif_idxs:
                        if motif_idx == 0:
                            if standardize_local is True:
                                matrix = add_A_all_norm[df_idx,:,:].T
                            else:
                                matrix = add_A.T
                            xticks = xticks_A
                            label = 'A'
                            L = np.arange(0,motif_A_len,1)
                            P = np.arange(0,motif_A_len+fig_pad*2,1)
                        elif motif_idx == 1:
                            if standardize_local is True:
                                matrix = add_B_all_norm[df_idx,:,:].T
                            else:
                                matrix = add_B.T
                            xticks = xticks_B
                            label = 'B'
                            L = np.arange(motif_A_len,motif_A_len+motif_B_len,1)
                            P = np.arange(0,motif_B_len,1)
                            P = P + 0.5
                        saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (label,gauge))
                        if not os.path.exists(saveDir_add):
                            os.makedirs(saveDir_add)
                        savePath = os.path.join(saveDir_add, 'mavenn_additive_%s_%s' % (label, df_idx))
                        squid_figs.matrix_and_logo(matrix, xticks, clim, label, L, P, alphabet, savePath,
                                                plot_matrix=False,
                                                plot_logo=True,
                                                ylabel='Additive effect')

                    if compare is True:
                        if 'ISM_single' in comparison_methods:
                            # evaluate min/max values for colormap
                            if np.abs(np.amin(np.array(ISM_single[start_full:stop_full]))) < np.abs(np.amax(np.array(ISM_single[start_full:stop_full]))):
                                clim = np.abs(np.amax(np.array(ISM_single[start_full:stop_full])))
                            else:
                                clim = np.abs(np.amin(np.array(ISM_single[start_full:stop_full])))
                            for motif_idx in motif_idxs:
                                if motif_idx == 0:
                                    if standardize_local is True:
                                        matrix = ISM_A_all_norm[df_idx,:,:].T
                                    else:
                                        matrix = ISM_A.T
                                    xticks = xticks_A
                                    label = 'A'
                                    L = np.arange(0,motif_A_len,1)
                                    P = np.arange(0,motif_A_len+fig_pad*2,1)
                                elif motif_idx == 1:
                                    matrix = ISM_B.T
                                    xticks = xticks_B
                                    label = 'B'
                                    L = np.arange(motif_A_len,motif_A_len+motif_B_len,1)
                                    P = np.arange(0,motif_B_len,1)
                                    P = P + 0.5
                                saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (label,gauge))
                                savePath = os.path.join(saveDir_add, 'ISM_%s_%s' % (label, df_idx))
                                squid_figs.matrix_and_logo(matrix, xticks, clim, label, L, P, alphabet, savePath,
                                                        plot_matrix=False,
                                                        plot_logo=True,
                                                        ylabel='ISM')
                            
                        if 'saliency' in comparison_methods or 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                            # evaluate min/max values for colormap
                            if np.abs(np.amin(np.array(other[start_full:stop_full]))) < np.abs(np.amax(np.array(other[start_full:stop_full]))):
                                clim = np.abs(np.amax(np.array(other[start_full:stop_full])))
                            else:
                                clim = np.abs(np.amin(np.array(other[start_full:stop_full])))
                            for motif_idx in motif_idxs:
                                if motif_idx == 0:
                                    if standardize_local is True:
                                        matrix = sal_A_all_norm[df_idx,:,:].T
                                    else:
                                        matrix = sal_A.T
                                    xticks = xticks_A
                                    label = 'A'
                                    L = np.arange(0,motif_A_len,1)
                                    P = np.arange(0,motif_A_len+fig_pad*2,1)
                                elif motif_idx == 1:
                                    if standardize_local is True:
                                        matrix = sal_B_all_norm[df_idx,:,:].T
                                    else:
                                        matrix = sal_B.T
                                    xticks = xticks_B
                                    label = 'B'
                                    L = np.arange(motif_A_len,motif_A_len+motif_B_len,1)
                                    P = np.arange(0,motif_B_len,1)
                                    P = P + 0.5
                                    
                                saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (label,gauge))
                                if 'saliency' in comparison_methods:
                                    savePath = os.path.join(saveDir_add, 'saliency_%s_%s' % (label, df_idx))
                                    squid_figs.matrix_and_logo(matrix, xticks, clim, label, L, P, alphabet, savePath,
                                                            plot_matrix=False,
                                                            plot_logo=True,
                                                            ylabel='Saliency')
                                elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                                    savePath = os.path.join(saveDir_add, 'deepLIFT_%s_%s' % (label, df_idx))
                                    squid_figs.matrix_and_logo(matrix, xticks, clim, label, L, P, alphabet, savePath,
                                                            plot_matrix=False,
                                                            plot_logo=True,
                                                            ylabel='deepLIFT')
                    
                    # =============================================================================
                    # Plot pairwise A/B/I terms per sequence            
                    # =============================================================================
                    if gpmap == 'pairwise':
                        # evaluate min/max values for colormap
                        if np.abs(np.amin(mavenn_pairwise)) < np.abs(np.amax(mavenn_pairwise)):
                            clim = np.abs(np.amax(mavenn_pairwise))
                        else:
                            clim = np.abs(np.amin(mavenn_pairwise))
                    
                        for i in [0,1,2]:
                            if i == 0: #plot motif_A pairwise energy (LHS)
                                pw_matrix = pw_A
                                pw_xticks = xticks_A
                                pw_label = 'A'
                            elif i == 1: #plot motif_B pairwise energy (RHS)
                                pw_matrix = pw_B
                                pw_xticks = xticks_B
                                pw_label = 'B'
                            elif i == 2: #plot motif_A/motif_B pairwise interaction energy (center)
                                pw_matrix = pw_I
                                pw_xticks = xticks_I
                                pw_label = 'I'
                        
                            fig, ax = plt.subplots(figsize=[10,5])
                            ax, cb = mavenn.heatmap_pairwise(values=pw_matrix,
                                                            alphabet=alphabet,
                                                            ax=ax,
                                                            gpmap_type='pairwise',
                                                            cmap_size='2%',
                                                            show_alphabet=True,
                                                            alphabet_size=7,
                                                            cmap='seismic',
                                                            clim=[-1.*clim, clim],
                                                            cmap_pad=.35,
                                                            show_seplines=True,
                                                            sepline_kwargs = {'color': 'k',
                                                                            'linestyle': '-',
                                                                            'linewidth': .5})
                            ax.xaxis.set_ticks(np.arange(0,pw_matrix.shape[0],1))
                            ax.set_xticklabels(pw_xticks, fontsize=7)
                            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                            cb.set_label(r'Pairwise Effect',
                                        labelpad=8, ha='center', va='center', rotation=-90)
                            cb.outline.set_visible(False)
                            cb.ax.tick_params(direction='in', size=20, color='white')
                            plt.tight_layout()
                            saveDir_pw = os.path.join(saveDir,'PW_%s' % pw_label)
                            if not os.path.exists(saveDir_pw):
                                os.makedirs(saveDir_pw)
                            savePath = os.path.join(saveDir_pw, 'mavenn_pairwise_%s_%s' % (pw_label, df_idx))
                            plt.savefig(savePath, facecolor='w', dpi=200)
                            plt.close()
                        

    # =============================================================================
    # Plot average of additive A/B terms over all sequences
    # =============================================================================
    if filter_muts is True:
        add_A_all = add_A_all[filter_muts_A]
        ISM_A_all = ISM_A_all[filter_muts_A]
        sal_A_all = sal_A_all[filter_muts_A]
        if scope == 'inter':
            add_B_all = add_B_all[filter_muts_B]
            if compare is True:
                ISM_B_all = ISM_B_all[filter_muts_B]
                sal_B_all = sal_B_all[filter_muts_B]

    if standardize_local is True:
        add_A_mean = np.mean(add_A_all_norm, axis=0)
    else:
        add_A_mean = np.mean(add_A_all, axis=0)
    if scope == 'inter':
        if standardize_local is True:
            add_B_mean = np.mean(add_B_all_norm, axis=0)
        else:
            add_B_mean = np.mean(add_B_all, axis=0)
    if compare is True:
        if standardize_local is True:
            ISM_A_mean = np.mean(ISM_A_all_norm, axis=0)
            sal_A_mean = np.mean(sal_A_all_norm, axis=0)
        else:
            ISM_A_mean = np.mean(ISM_A_all, axis=0)
            sal_A_mean = np.mean(sal_A_all, axis=0)
        if scope == 'inter':
            if standardize_local is True:
                ISM_B_mean = np.mean(ISM_B_all_norm, axis=0)
                sal_B_mean = np.mean(sal_B_all_norm, axis=0)
            else:
                ISM_B_mean = np.mean(ISM_B_all, axis=0)
                sal_B_mean = np.mean(sal_B_all, axis=0)


    # save stack of attribution maps fo file
    saveDir_add = os.path.join(saveDir,'ADD_A/ADD_%s' % gauge)
    if not os.path.exists(saveDir_add):
        os.makedirs(saveDir_add)
    np.save(os.path.join(saveDir_add, 'all_add_A.npy'), add_A_all)
    if compare is True:
        np.save(os.path.join(saveDir_add, 'all_ISM_A.npy'), ISM_A_all)
        np.save(os.path.join(saveDir_add, 'all_other_A.npy'), sal_A_all)
    if standardize_local is True:
        np.save(os.path.join(saveDir_add, 'all_norm_add_A.npy'), add_A_all_norm)
        if compare is True:
            np.save(os.path.join(saveDir_add, 'all_norm_ISM_A.npy'), ISM_A_all_norm)
            np.save(os.path.join(saveDir_add, 'all_norm_other_A.npy'), sal_A_all_norm)
        

    # find values to create global colormap
    if scope == 'intra':
        global_min = np.amin(add_A_mean)
        global_max = np.amax(add_A_mean)
    if scope == 'inter':
        add_A_min = np.amin(add_A_mean)
        add_B_min = np.amin(add_B_mean)
        global_min = np.amin([add_A_min,add_B_min])  
        add_A_max = np.amax(add_A_mean)
        add_B_max = np.amax(add_B_mean)
        global_max = np.amax([add_A_max,add_B_max])
    theta_limit = np.amax([np.abs(global_min), np.abs(global_max)])
        
    if np.abs(global_min) < np.abs(global_max):
        clim = np.abs(global_max)
    else:
        clim = np.abs(global_min)

    for motif_idx in motif_idxs:
        if motif_idx == 0:
            add_matrix_all = add_A_mean.T
            add_label = 'A'
            L = np.arange(0,motif_A_len+fig_pad*2,1)
            P = np.arange(0,motif_A_len+fig_pad*2,1)
            add_pw_all = np.mean(pw_A_all, axis=0)
        elif motif_idx == 1:
            add_matrix_all = add_B_mean.T
            add_label = 'B'
            L = np.arange(0,motif_B_len+fig_pad*2,1)
            P = np.arange(0,motif_B_len+fig_pad*2,1)
            add_pw_all = np.mean(pw_B_all, axis=0)
        saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (add_label,gauge))     
        if not os.path.exists(saveDir_add):
            os.makedirs(saveDir_add)
        savePath = os.path.join(saveDir_add, 'avgLogo_additive_%s' % add_label)
        squid_figs.matrix_and_logo(add_matrix_all, None, clim, add_label, L, P, alphabet, savePath,
                                plot_matrix=False,
                                plot_logo=True,
                                ylabel='Additive effect')
        if 1:
            add_matrix_all = squid_utils.arr2pd(add_matrix_all.T, alphabet)
            add_matrix_all.to_csv(os.path.join(saveDir_add, 'avg_additive_%s.csv' % add_label))
            np.save(os.path.join(saveDir_add, 'avg_additive_%s.npy' % add_label), add_matrix_all)

            if 1: #plot additive matrix
                divnorm=colors.TwoSlopeNorm(vmin=-1.*theta_limit, vcenter=0., vmax=theta_limit)
                fig, ax = plt.subplots(figsize=[7,1.5])#fig, ax = plt.subplots(figsize=[7,1.5])
                im = plt.pcolormesh(add_matrix_all.T,
                                    norm=divnorm,
                                    edgecolors='k',
                                    linewidth=.35,
                                    cmap='seismic',
                                    color='gray')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='3.0%', pad=0.15)
                ax.set_aspect('equal')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                B = ['A', 'C', 'G', 'T']
                ax.set_yticks([0.5, 1.5, 2.5, 3.5])
                ax.set_yticklabels(B, fontsize=12)
                plt.colorbar(im, cmap='seismic', cax=cax)
                plt.clim(-1.*theta_limit, theta_limit)

                plt.tight_layout()
                if 1:
                    plt.savefig(os.path.join(saveDir_add, 'avgMatrix_additive_%s.pdf' % add_label), facecolor='w', dpi=600)
                    plt.close()
                else:
                    plt.show()

        
    if compare is True:
        for motif_idx in motif_idxs:
            if motif_idx == 0:
                add_matrix_all = ISM_A_mean.T
                add_label = 'A'
                L = np.arange(0,motif_A_len+fig_pad*2,1)
                P = np.arange(0,motif_A_len+fig_pad*2,1)
            elif motif_idx == 1:
                add_matrix_all = ISM_B_mean.T
                add_label = 'B'
                L = np.arange(0,motif_B_len+fig_pad*2,1)
                P = np.arange(0,motif_B_len+fig_pad*2,1)
            saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (add_label, gauge))
            savePath = os.path.join(saveDir_add, 'avgLogo_ISM_%s' % add_label)
            squid_figs.matrix_and_logo(add_matrix_all, None, clim, add_label, L, P, alphabet, savePath,
                                    plot_matrix=False,
                                    plot_logo=True,
                                    ylabel='ISM')
        if 1:
            add_matrix_all = squid_utils.arr2pd(add_matrix_all.T, alphabet)
            add_matrix_all.to_csv(os.path.join(saveDir_add, 'avg_ISM_%s.csv' % add_label))
            np.save(os.path.join(saveDir_add, 'avg_ISM_%s.npy' % add_label), add_matrix_all)
            
        for motif_idx in motif_idxs:
            if motif_idx == 0:
                add_matrix_all = sal_A_mean.T
                add_label = 'A'
                L = np.arange(0,motif_A_len+fig_pad*2,1)
                P = np.arange(0,motif_A_len+fig_pad*2,1)
            elif motif_idx == 1:
                add_matrix_all = sal_B_mean.T
                add_label = 'B'
                L = np.arange(0,motif_B_len+fig_pad*2,1)
                P = np.arange(0,motif_B_len+fig_pad*2,1)
            saveDir_add = os.path.join(saveDir,'ADD_%s/ADD_%s' % (add_label, gauge))
            if 'saliency' in comparison_methods:
                savePath = os.path.join(saveDir_add, 'avgLogo_other_%s' % add_label)
                squid_figs.matrix_and_logo(add_matrix_all, None, clim, add_label, L, P, alphabet, savePath,
                                        plot_matrix=False,
                                        plot_logo=True,
                                        ylabel='Saliency')
                if 1:
                    add_matrix_all = squid_utils.arr2pd(add_matrix_all.T, alphabet)
                    add_matrix_all.to_csv(os.path.join(saveDir_add, 'avg_other_%s.csv' % add_label))
            elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                savePath = os.path.join(saveDir_add, 'avglogo_other_%s' % add_label)
                squid_figs.matrix_and_logo(add_matrix_all, None, clim, add_label, L, P, alphabet, savePath,
                                        plot_matrix=False,
                                        plot_logo=True,
                                        ylabel='deepLIFT')
                if 1:
                    add_matrix_all = squid_utils.arr2pd(add_matrix_all.T, alphabet)
                    add_matrix_all.to_csv(os.path.join(saveDir_add, 'avg_other_%s.csv' % add_label))
                    np.save(os.path.join(saveDir_add, 'avg_other_%s.npy' % add_label), add_matrix_all)

    # =============================================================================
    # Plot average of pairwise A/B/I terms over all sequences
    # =============================================================================
    if gpmap == 'pairwise':
        if filter_muts is True:
            pw_A_all = pw_A_all[filter_muts_A]
            if scope == 'inter':
                pw_B_all = pw_B_all[filter_muts_B]
                
        pw_A_mean = np.mean(pw_A_all, axis=0)
        if scope == 'inter':
            pw_B_mean = np.mean(pw_B_all, axis=0)
            pw_I_mean = np.mean(pw_I_all, axis=0)
        
        # find values to create global colormap
        if scope == 'intra':
            global_min = np.amin(pw_A_mean)
            global_max = np.amax(pw_A_mean)
        if scope == 'inter':
            pw_A_min = np.amin(pw_A_mean)
            pw_B_min = np.amin(pw_B_mean)
            pw_I_min = np.amin(pw_I_mean)
            global_min = np.amin([pw_A_min,pw_B_min,pw_I_min])  
            pw_A_max = np.amax(pw_A_mean)
            pw_B_max = np.amax(pw_B_mean)
            pw_I_max = np.amax(pw_I_mean)
            global_max = np.amax([pw_A_max,pw_B_max,pw_I_max])
        
        if np.abs(global_min) < np.abs(global_max):
            clim = np.abs(global_max)
        else:
            clim = np.abs(global_min)
        
        for i in [0,1,2]:
            if i == 0: #plot motif_A pairwise energy (LHS)
                pw_matrix_all = pw_A_mean
                pw_label = 'A'
            elif i == 1: #plot motif_B pairwise energy (RHS)
                pw_matrix_all = pw_B_mean
                pw_label = 'B'
            elif i == 2: #plot motif_A/motif_B pairwise interaction energy (center)
                pw_matrix_all = pw_I_mean
                pw_label = 'I'
                
            fig, ax = plt.subplots(figsize=[10,5])
            ax, cb = mavenn.heatmap_pairwise(values=pw_matrix_all,
                                            alphabet=alphabet,
                                            ax=ax,
                                            gpmap_type='pairwise',
                                            cmap_size='2%',
                                            show_alphabet=True,
                                            alphabet_size=7,
                                            cmap='seismic',
                                            clim=[-1.*clim, clim],
                                            cmap_pad=.35,
                                            show_seplines=True,
                                            sepline_kwargs = {'color': 'k',
                                                            'linestyle': '-',
                                                            'linewidth': .5})
            if i == 0:
                ax.xaxis.set_ticks(np.arange(0,motif_A_len,1))
                ax.set_xticklabels(np.arange(motif_A_len), fontsize=7)
            elif i == 1:
                ax.xaxis.set_ticks(np.arange(0,motif_B_len,1))
                ax.set_xticklabels(np.arange(motif_A_len,motif_A_len+motif_B_len), fontsize=7)
            elif i == 2:
                ax.xaxis.set_ticks(np.arange(0,pw_matrix_all.shape[0],1))
            cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cb.set_label(r'Pairwise effect',
                        labelpad=8, ha='center', va='center', rotation=-90)
            cb.outline.set_visible(False)
            cb.ax.tick_params(direction='in', size=20, color='white')
            plt.tight_layout()
            saveDir_pw = os.path.join(saveDir,'PW_%s' % pw_label)
            if not os.path.exists(saveDir_pw):
                os.makedirs(saveDir_pw)
            savePath = os.path.join(saveDir_pw, '_mavenn_pairwise_%s_avg.pdf' % (pw_label))
            plt.savefig(savePath, facecolor='w', dpi=600)
            plt.close()
            np.save(os.path.join(saveDir_pw, '_mavenn_pairwise_%s_avg.npy' % (pw_label)), pw_matrix_all)
            
            
    # =============================================================================
    # Plot collection of figures comparing statistics across attribution maps
    # =============================================================================
    if 1:
        if compare is False: #fill variables with repeated information to progress
            ISM_A_all = add_A_all
            ISM_A_all_norm = add_A_all_norm
            ISM_A_mean = add_A_mean
            sal_A_all = add_A_all
            sal_A_all_norm = add_A_all_norm
            sal_A_mean = add_A_mean
            if max_flank is None:
                print("Creating (duplicate) box plots, since 'compare' set to False...")
        else:
            if max_flank is None:
                print("Creating box plots...")

        if standardize_local is False and standardize_global is False:
            add_A_std = np.std(add_A_all, axis=0)
            add_A_singles = np.linalg.norm(add_A_all - add_A_mean, axis=(1,2))
            if compare is True:
                ISM_A_std = np.std(ISM_A_all, axis=0)
                sal_A_std = np.std(sal_A_all, axis=0)
                ISM_A_singles = np.linalg.norm(ISM_A_all - ISM_A_mean, axis=(1,2))
                sal_A_singles = np.linalg.norm(sal_A_all - sal_A_mean, axis=(1,2))
            if scope == 'inter':
                add_B_std = np.std(add_B_all, axis=0)
                add_B_singles = np.linalg.norm(add_B_all - add_B_mean, axis=(1,2))
                if compare is True:
                    ISM_B_std = np.std(ISM_B_all, axis=0)
                    sal_B_std = np.std(sal_B_all, axis=0)
                    ISM_B_singles = np.linalg.norm(ISM_B_all - ISM_B_mean, axis=(1,2))
                    sal_B_singles = np.linalg.norm(sal_B_all - sal_B_mean, axis=(1,2))

        if standardize_local is True and standardize_global is False:
            add_A_std = np.std(add_A_all_norm, axis=0)
            add_A_singles = np.linalg.norm(add_A_all_norm - add_A_mean, axis=(1,2))
            if compare is True:
                ISM_A_std = np.std(ISM_A_all_norm, axis=0)
                sal_A_std = np.std(sal_A_all_norm, axis=0)
                ISM_A_singles = np.linalg.norm(ISM_A_all_norm - ISM_A_mean, axis=(1,2))
                sal_A_singles = np.linalg.norm(sal_A_all_norm - sal_A_mean, axis=(1,2))
            if scope == 'inter':
                add_B_std = np.std(add_B_all_norm, axis=0)
                add_B_singles = np.linalg.norm(add_B_all_norm - add_B_mean, axis=(1,2))
                if compare is True:
                    ISM_B_std = np.std(ISM_B_all_norm, axis=0)
                    sal_B_std = np.std(sal_B_all_norm, axis=0)
                    ISM_B_singles = np.linalg.norm(ISM_B_all_norm - ISM_B_mean, axis=(1,2))
                    sal_B_singles = np.linalg.norm(sal_B_all_norm - sal_B_mean, axis=(1,2))

        elif standardize_local is True and standardize_global is True:
            add_A_all_norm /= LA.norm(add_A_all_norm)
            add_A_std = np.std(add_A_all_norm, axis=0)
            add_A_singles = np.linalg.norm(add_A_all_norm - add_A_mean, axis=(1,2))
            if compare is True:
                ISM_A_all_norm /= LA.norm(ISM_A_all_norm)
                sal_A_all_norm /= LA.norm(sal_A_all_norm)
                ISM_A_std = np.std(ISM_A_all_norm, axis=0)
                sal_A_std = np.std(sal_A_all_norm, axis=0)
                ISM_A_singles = np.linalg.norm(ISM_A_all_norm - ISM_A_mean, axis=(1,2))
                sal_A_singles = np.linalg.norm(sal_A_all_norm - sal_A_mean, axis=(1,2))
            if scope == 'inter':
                add_B_all_norm /= LA.norm(add_B_all_norm)
                add_B_std = np.std(add_B_all_norm, axis=0)
                add_B_singles = np.linalg.norm(add_B_all_norm - add_B_mean, axis=(1,2))
                if compare is True:
                    ISM_B_all_norm /= LA.norm(ISM_B_all_norm)
                    sal_B_all_norm /= LA.norm(sal_B_all_norm)
                    ISM_B_std = np.std(ISM_B_all_norm, axis=0)
                    sal_B_std = np.std(sal_B_all_norm, axis=0)
                    ISM_B_singles = np.linalg.norm(ISM_B_all_norm - ISM_B_mean, axis=(1,2))
                    sal_B_singles = np.linalg.norm(sal_B_all_norm - sal_B_mean, axis=(1,2))

        elif standardize_local is False and standardize_global is True:
            """
            Note:   Not a fair comparison, since mavenn outputs arrive normalized..
                    while the other attribution methods do not
            """
            add_A_all /= LA.norm(add_A_all)
            add_A_std = np.std(add_A_all, axis=0)
            add_A_singles = np.linalg.norm(add_A_all - add_A_mean, axis=(1,2))
            if compare is True:
                ISM_A_all /= LA.norm(ISM_A_all)
                sal_A_all /= LA.norm(sal_A_all)
                ISM_A_std = np.std(ISM_A_all, axis=0)
                sal_A_std = np.std(sal_A_all, axis=0)
                ISM_A_singles = np.linalg.norm(ISM_A_all - ISM_A_mean, axis=(1,2))
                sal_A_singles = np.linalg.norm(sal_A_all - sal_A_mean, axis=(1,2))
            if scope == 'inter':
                add_B_all /= LA.norm(add_B_all)
                add_B_std = np.std(add_B_all, axis=0)
                add_B_singles = np.linalg.norm(add_B_all - add_B_mean, axis=(1,2))
                if compare is True:
                    ISM_B_all /= LA.norm(ISM_B_all)
                    sal_B_all /= LA.norm(sal_B_all)
                    ISM_B_std = np.std(ISM_B_all, axis=0)
                    sal_B_std = np.std(sal_B_all, axis=0)
                    ISM_B_singles = np.linalg.norm(ISM_B_all - ISM_B_mean, axis=(1,2))
                    sal_B_singles = np.linalg.norm(sal_B_all - sal_B_mean, axis=(1,2))
            
        for motif_idx in motif_idxs:
            if motif_idx == 0:
                add_all = add_A_all
                add_pad = add_A_all_norm
                add_std = add_A_std
                add_singles = add_A_singles
                if compare is True:
                    ISM_all = ISM_A_all
                    ISM_pad = ISM_A_all_norm
                    ISM_std = ISM_A_std
                    ISM_singles = ISM_A_singles
                    sal_all = sal_A_all
                    sal_pad = sal_A_all_norm
                    sal_std = sal_A_std
                    sal_singles = sal_A_singles
                label = 'A'
                motif = motif_A
                motif_pad = fig_pad*'N' + motif + fig_pad*'N'
                motif_filter = squid_utils.seq2oh(motif_pad, alphabet)
                WT_seqs = WT_seqs_A
            elif motif_idx == 1:
                add_all = add_B_all
                add_pad = add_B_all_norm
                add_std = add_B_std
                add_singles = add_B_singles
                if compare is True:
                    ISM_all = ISM_B_all
                    ISM_pad = ISM_B_all_norm
                    ISM_std = ISM_B_std
                    ISM_singles = ISM_B_singles
                    sal_all = sal_B_all
                    sal_pad = sal_B_all_norm
                    sal_std = sal_B_std
                    sal_singles = sal_B_singles
                label = 'B'
                motif = motif_B
                motif_pad = fig_pad*'N' + motif + fig_pad*'N'
                motif_filter = squid_utils.seq2oh(motif_pad, alphabet)
                WT_seqs = WT_seqs_B
                
            # =============================================================================
            # Boxplot comparison of Euclidean distances for gold-standard attribution maps and additive models
            # =============================================================================
            if 'saliency' in comparison_methods and compare is True:
                all_singles = {'ISM':ISM_singles, 'SAL':sal_singles, 'ADD':add_singles}
            elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods and compare is True:    
                all_singles = {'ISM':ISM_singles, 'DL':sal_singles, 'ADD':add_singles}
            else:
                all_singles = {'n/a':add_singles*0, 'n/a':add_singles*0, 'ADD':add_singles}
            
            if max_flank == None:
                print('Boxplot statistics:')
                if compare is True:
                    print('Attribution method 1 mean: ', round(np.mean(ISM_singles),2))
                    print('Attribution method 2 mean:', round(np.mean(sal_singles),2))
                print('SQUID mean:', round(np.mean(add_singles),2))
            else:
                saveDir2 = os.path.join(userDir, 'd_outputs_analysis/%s/' % (model_name))
                '''
                    if running a loop via 4_batch.sh, save the mean error for each attribution method to an array; 
                    used in figures showing mean-attribution-errors as a function of flanking nucleotides
                '''

                if flank_idx == 0:
                    meanErrors = np.zeros(shape=(3,max_flank+1)) #i.e., {3 attribution methods; max allowed flanking options}
                    meanErrors[0,flank_idx] = round(np.mean(ISM_singles),2)
                    meanErrors[1,flank_idx] = round(np.mean(sal_singles),2)
                    meanErrors[2,flank_idx] = round(np.mean(add_singles),2)
                    np.save(os.path.join(saveDir2,'SQUID_%s_%s_mut%s/errors_%s.npy' % (motif_A_name, scope, use_mut, gauge)), meanErrors)
                    
                    # go ahead and create array here for the p-value loop (see below)
                    meanErrorsPvals = np.zeros(shape=(3,max_flank+1)) #i.e., {3 P-val combinations; max allowed flanking options}

                elif flank_idx != 0:
                    meanErrors = np.load(os.path.join(saveDir2,'SQUID_%s_%s_mut%s/errors_%s.npy' % (motif_A_name, scope, use_mut, gauge)))
                    meanErrors[0,flank_idx] = round(np.mean(ISM_singles),2)
                    meanErrors[1,flank_idx] = round(np.mean(sal_singles),2)
                    meanErrors[2,flank_idx] = round(np.mean(add_singles),2)
                    np.save(os.path.join(saveDir2,'SQUID_%s_%s_mut%s/errors_%s.npy' % (motif_A_name, scope, use_mut, gauge)), meanErrors)

                    meanErrorsPvals = np.load(os.path.join(saveDir2,'SQUID_%s_%s_mut%s/errors_%s_Pvals.npy' % (motif_A_name, scope, use_mut, gauge)))

                #if flank_idx == max_flank:
                    #print('Mean-attribution-error matrix:\n', meanErrors)

            if 0:
                print('ISM argmax',np.argmax(ISM_singles))
                print('Other argmax',np.argmax(sal_singles))
                print('ADD argmax',np.argmax(add_singles))

            
            fig = plt.figure()#figsize=(4,3))
            flierprops = dict(marker='^', markerfacecolor='green', markersize=14, linestyle='none')
            box = plt.boxplot(all_singles.values(), showfliers=False, showmeans=True, meanprops=flierprops)
            
            # add MWU information to plot
            if compare is True:
                ls = list(range(1, 4))
                combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
                ax_bottom, ax_top = plt.gca().get_ylim()
                y_range = ax_top - ax_bottom
                level_idx = 3
                pval_idx = 0
                for x1, x2 in combinations:#range(len(comparison_methods)):
                    if x1 == 1 and x2 == 3:
                        mwu_stat, pval = scipy.stats.mannwhitneyu(add_singles, ISM_singles, alternative='less')
                    elif x1 == 1 and x2 == 2:
                        mwu_stat, pval = scipy.stats.mannwhitneyu(sal_singles, ISM_singles, alternative='less')
                    elif x1 == 2 and x2 == 3:
                        mwu_stat, pval = scipy.stats.mannwhitneyu(add_singles, sal_singles, alternative='less')


                    if max_flank is not None:
                        if pval_idx == 0:
                            meanErrorsPvals[pval_idx,flank_idx] = pval
                            pval_idx += 1
                        elif pval_idx == 1:
                            meanErrorsPvals[pval_idx,flank_idx] = pval
                            pval_idx += 1
                        elif pval_idx == 2:
                            meanErrorsPvals[pval_idx,flank_idx] = pval
                    else:
                        #print('MWU statistic (%s–%s): %s' % (x1,x2,mwu_stat))
                        print('MWU p-value (%s–%s): %s' % (x1,x2,pval))

                    if pval < 0.001:
                        sig_symbol = '***'
                    elif pval < 0.01:
                        sig_symbol = '**'
                    elif pval < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = 'ns'
                    bar_height = (y_range * 0.07 * level_idx) + ax_top
                    bar_tips = bar_height - (y_range * 0.02)
                    plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
                    text_height = bar_height*.95 + (y_range * 0.001)
                    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
                    level_idx -= 1

                if max_flank is not None:
                    np.save(os.path.join(saveDir2,'SQUID_%s_%s_mut%s/errors_%s_Pvals.npy' % (motif_A_name, scope, use_mut, gauge)), meanErrorsPvals)
                    #if flank_idx == max_flank:
                        #print('Mean-attribution-error Pvals:\n', meanErrorsPvals)

                # annotate sample size below each box
                for c in ls:
                    plt.gca().text(c, 0 + y_range * 0.02, r'$n = %s$' % seq_total, ha='center', size='x-small')
                        
                plt.title('%s | %s core mutations' % (motif_A_name, use_mut) + '\n' + r'$\ast\ast\ast: \alpha < 0.001$ | $\ast\ast: \alpha < 0.01$ | $\ast: \alpha < 0.05$')
                if 'saliency' in comparison_methods:
                    plt.xticks([1, 2, 3], ['ISM', 'SAL', 'ADD'], rotation=40, fontsize=12, ha='right')
                elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                    plt.xticks([1, 2, 3], ['ISM', 'DL', 'ADD'], rotation=40, fontsize=12, ha='right')
                else:
                    plt.xticks([1, 2, 3], ['ISM', 'n/a', 'ADD'], rotation=40, fontsize=12, ha='right')
                ax = plt.gca()
                
                for singles_i in [1,2,3]:
                    singles_y = list(all_singles.values())[singles_i-1]
                    singles_x = np.random.normal(singles_i, 0.04, size=len(singles_y))
                    ax.plot(singles_x, singles_y, 'r.', alpha=0.2)
                
                plt.setp(ax.get_yticklabels(),fontsize=12);
                plt.ylabel('Error', fontsize=12)
                plt.ylim(0,plt.gca().get_ylim()[1])
                plt.tight_layout()
                
                saveDir_stats = os.path.join(saveDir,'stats/stats_%s' % gauge)
                if not os.path.exists(saveDir_stats):
                    os.makedirs(saveDir_stats)
                plt.savefig(os.path.join(saveDir_stats, 'compare_boxplot_%s' % label), facecolor='w', dpi=200)
                plt.close()
                
                np.save(os.path.join(saveDir_stats, 'compare_boxplot_%s_values.npy' % label), all_singles)

                # =============================================================================
                # Barplot comparison of standard deviations for gold-standard attribution maps and additive models
                # =============================================================================
                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
                if standardize_local is True or standardize_global is True:
                    max_std = np.amax([ISM_std.max(), sal_std.max(), add_std.max()])
                    axes[0].set_ylim(0,max_std)
                    axes[1].set_ylim(0,max_std)
                    axes[2].set_ylim(0,max_std)
                axes[0].set_title('%s core mutations' % use_mut)
                axes[0].set_ylabel('ISM', labelpad=10)
                squid_utils.arr2pd(ISM_std, alphabet).plot(kind='bar', ax=axes[0]).legend(loc='upper right')
                if 'saliency' in comparison_methods:
                    axes[1].set_ylabel('Saliency', labelpad=10)
                elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                    axes[1].set_ylabel('deepLIFT', labelpad=10)
                squid_utils.arr2pd(sal_std, alphabet).plot(kind='bar', ax=axes[1]).legend(loc='upper right')
                axes[2].set_ylabel('Additive', labelpad=10)
                squid_utils.arr2pd(add_std, alphabet).plot(kind='bar', ax=axes[2]).legend(loc='upper right')
                plt.tight_layout()
                saveDir_stats = os.path.join(saveDir,'stats/stats_%s' % gauge)
                if not os.path.exists(saveDir_stats):
                    os.makedirs(saveDir_stats)
                plt.savefig(os.path.join(saveDir_stats, 'compare_std_barplot_%s' % label), facecolor='w', dpi=200)
                plt.close()
            
                # =============================================================================
                # Compare standard deviations for specific consensus-motif basepairs
                # =============================================================================
                ISM_std_core = np.diag(np.dot(ISM_std, motif_filter.T))
                sal_std_core = np.diag(np.dot(sal_std, motif_filter.T))
                add_std_core = np.diag(np.dot(add_std, motif_filter.T))
                std_core = np.vstack((ISM_std_core, sal_std_core, add_std_core))
                if 'saliency' in comparison_methods:
                    labels = {'ISM': std_core[0,:],
                            'Saliency': std_core[1,:],
                            'Additive': std_core[2,:]}
                if 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                    labels = {'ISM': std_core[0,:],
                            'deepLIFT': std_core[1,:],
                            'Additive': std_core[2,:]}
                else:
                    labels = {'ISM': std_core[0,:],
                            'n/a': std_core[1,:]*0,
                            'Additive': std_core[2,:]}
                    
                std_core = pd.DataFrame.from_dict(labels, orient='index').T
                fig, ax = plt.subplots(figsize=(10,5))
                ax.set_title('%s core mutations |  $n$ = %s' % (use_mut, seq_total))
                std_core.plot(kind='bar', ax=ax).legend(loc='upper right')
                ax.set_ylabel('standard deviation', labelpad=10)
                ax.xaxis.set_ticks(np.arange(0,len(motif_pad),1))
                labels = [item.get_text() for item in ax.get_xticklabels()]
                for count, value in enumerate(motif_pad):
                    labels[count] = value
                ax.set_xticklabels(labels, rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(saveDir_stats, 'compare_std_core_%s.png' % label), facecolor='w', dpi=200)
                plt.close()
            
                # =============================================================================
                # Plot success rates for specific consensus-motif basepairs
                # =============================================================================
                if 0:
                    ISM_freq = np.zeros((add_all.shape[0], len(motif_pad)))
                    sal_freq = np.zeros((add_all.shape[0], len(motif_pad)))
                    add_freq = np.zeros((add_all.shape[0], len(motif_pad)))
                    for seq in range(add_all.shape[0]):
                        ISM_temp = np.diag(np.dot(ISM_all[seq], motif_filter.T))
                        ISM_temp = np.maximum(0, ISM_temp)
                        ISM_freq[seq,:] = [1 if a_ > 0 else a_ for a_ in ISM_temp]
                        sal_temp = np.diag(np.dot(sal_all[seq], motif_filter.T))
                        sal_temp = np.maximum(0, sal_temp)
                        sal_freq[seq,:] = [1 if a_ > 0 else a_ for a_ in sal_temp]
                        add_temp = np.diag(np.dot(add_all[seq], motif_filter.T))
                        add_temp = np.maximum(0, add_temp)
                        add_freq[seq,:] = [1 if a_ > 0 else a_ for a_ in add_temp]
                    ISM_prob = np.sum(ISM_freq, axis=0)/float(add_all.shape[0])
                    sal_prob = np.sum(sal_freq, axis=0)/float(add_all.shape[0])
                    add_prob = np.sum(add_freq, axis=0)/float(add_all.shape[0])
                    prob_core = np.vstack((ISM_prob, sal_prob, add_prob))
                    if 'saliency' in comparison_methods:
                        labels = {'ISM': prob_core[0,:],
                                'Saliency': prob_core[1,:],
                                'Additive': prob_core[2,:]}
                    elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                        labels = {'ISM': prob_core[0,:],
                                'deepLIFT': prob_core[1,:],
                                'Additive': prob_core[2,:]}
                    else:
                        labels = {'ISM': prob_core[0,:],
                                'n/a': prob_core[1,:]*0,
                                'Additive': prob_core[2,:]}
                        
                    prob_core = pd.DataFrame.from_dict(labels, orient='index').T
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.set_title('%s core mutations |  $n$ = %s' % (use_mut, add_all.shape[0]))
                    prob_core.plot(kind='bar', ax=ax).legend(loc='upper right')
                    ax.set_ylabel('success rate', labelpad=10)
                    ax.xaxis.set_ticks(np.arange(0,len(motif_pad),1))
                    labels = [item.get_text() for item in ax.get_xticklabels()]
                    for count, value in enumerate(motif_pad):
                        labels[count] = value
                    ax.set_xticklabels(labels, rotation=0)
                    plt.tight_layout()
                    plt.savefig(os.path.join(saveDir_stats, 'compare_success_core_%s.png' % label), facecolor='w', dpi=200)
                    plt.close()
                
            # =============================================================================
            # Plot close-up comparisons of motif attribution maps
            #       Orange text:   WT sequence
            #       Blue text:     Optimal motif core
            # =============================================================================
            gap = 10 #positive integer in the open interval: (1, 'seq_total')
            if gap > seq_total:
                gap = seq_total
            chunks = np.arange(0,seq_total,gap) #plot motif comparisons in groups of 'gap' per figure
            M = 4 #number of columns
            y = 1.25 #figure title height
            
            if show_compared is True:
                print('Generating close-up motif comparisons...')
                for chunk in chunks:
                    fig, axs = plt.subplots(10, M, figsize=[16,14])
                    for i in range(chunk,chunk+gap): #designed to display 'gap' subplots at a time                
                        for j in range(M):
                            if j == 0:
                                if i-chunk == 0:
                                    axs[i-chunk,j].set_title('Core mutants', fontsize=24, y=y, loc='center')
                                axs[i-chunk,j].text(0.4, 0.5, '%s' % use_mut, fontsize=24)
                                axs[i-chunk,j].axis('off')
                            
                            if j == 1:
                                if i-chunk == 0:
                                    axs[i-chunk,j].set_title('ISM', fontsize=24, y=y, loc='center')
                                ISM_matrix = squid_utils.arr2pd(ISM_pad[i], alphabet)
                                if 0: #classic representation
                                    logo = logomaker.Logo(ISM_matrix,
                                                        ax=axs[i-chunk,j], center_values=True, 
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='classic')
                                else: #enrichment representation
                                    logo = logomaker.Logo(ISM_matrix,
                                                        ax=axs[i-chunk,j], center_values=True,
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='dimgray')
                                    try:
                                        logo.style_glyphs_in_sequence(sequence=WT_seqs[i], color='darkorange')
                                    except:
                                        pass
                                    if 1:
                                        groups =[list(g) for _,g in groupby(range(len(motif_pad)),lambda idx:motif_pad[idx])]
                                        for g in groups:
                                            if motif_pad[g[0]] != 'N':
                                                logo.highlight_position_range(pmin=g[0], pmax=g[-1], zorder=-10, color='powderblue', alpha=.35)

                            if j == 2:
                                if i-chunk == 0:
                                    if 'saliency' in comparison_methods:
                                        axs[i-chunk,j].set_title('Saliency', fontsize=24, y=y, loc='center')    
                                    elif 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
                                        axs[i-chunk,j].set_title('deepLIFT', fontsize=24, y=y, loc='center')
                                sal_matrix = squid_utils.arr2pd(sal_pad[i], alphabet)
                                if 0: #classic representation
                                    logo = logomaker.Logo(sal_matrix,
                                                        ax=axs[i-chunk,j], center_values=True,
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='classic')
                                else: #enrichment representation
                                    logo = logomaker.Logo(sal_matrix,
                                                        ax=axs[i-chunk,j], center_values=True,
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='dimgray')
                                    try:
                                        logo.style_glyphs_in_sequence(sequence=WT_seqs[i], color='darkorange')
                                    except:
                                        pass
                                    if 1:
                                        groups =[list(g) for _,g in groupby(range(len(motif_pad)),lambda idx:motif_pad[idx])]
                                        for g in groups:
                                            if motif_pad[g[0]] != 'N':
                                                logo.highlight_position_range(pmin=g[0], pmax=g[-1], zorder=-10, color='powderblue', alpha=.35)
                
                            if j == 3:
                                if i-chunk == 0:
                                    axs[i-chunk,j].set_title('MAVE-NN', fontsize=24, y=y, loc='center')
                                add_matrix = squid_utils.arr2pd(add_pad[i], alphabet)
                
                                if 0: #classic representation
                                    logo = logomaker.Logo(add_matrix,
                                                        ax=axs[i-chunk,j], center_values=True,
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='classic')
                                else: #enrichment representation
                                    logo = logomaker.Logo(add_matrix,
                                                        ax=axs[i-chunk,j], center_values=True,
                                                        font_name='Arial Rounded MT Bold', fade_below=.5, shade_below=.5,
                                                        color_scheme='dimgray')
                                    try:
                                        logo.style_glyphs_in_sequence(sequence=WT_seqs[i], color='darkorange')
                                    except:
                                        pass
                                    if 1:
                                        groups =[list(g) for _,g in groupby(range(len(motif_pad)),lambda idx:motif_pad[idx])]
                                        for g in groups:
                                            if motif_pad[g[0]] != 'N':
                                                logo.highlight_position_range(pmin=g[0], pmax=g[-1], zorder=-10, color='powderblue', alpha=.35)
                
                            if j > 0:
                                logo.style_spines(visible=False)
                                axs[i-chunk,j].get_xaxis().set_visible(False)
                                axs[i-chunk,j].get_yaxis().set_visible(False)
                
                    plt.tight_layout()
                    plt.savefig(os.path.join(saveDir_stats, 'closeup_logos_%03d_to_%03d_%s.png' % (chunk, chunk+gap, label)), facecolor='w', dpi=200)
                    plt.close()


    if gpmap == 'pairwise':
        if 0:
            dist_ys = np.zeros(21)
            dist_count = np.zeros(21)
            for idx, d in enumerate(dist_x):
                dist_ys[d] += dist_y[idx]
                dist_count[d] += 1

            plt. close('all')
            fig, ax = plt.subplots()
            ax.scatter(dist_x, dist_y, zorder=-10, alpha=.25)
            dist_ys_mean = dist_ys / dist_count
            ax.plot(dist_ys_mean)
            ax.set_xlabel('inter-motif distance')
            #ax.set_ylabel('sum of inter-motif interaction parameters')
            plt.show()


if __name__ == '__main__':
    path1 = os.path.dirname(os.path.abspath(__file__))

    """
    flank_idx : INT >= 0
        Number of flanking nucleotides to consider during attribution error analysis
    max_flank : INT <= 100, if using parameters chosen in our workflow
        Maximum number of flanking nucleotides to consider if using batch loop (see '4_batch.sh')

    """

    if len(sys.argv) == 4: #batch mode
        flank_idx = int(sys.argv[1])
        max_flank = int(sys.argv[2])
        gauge_type = str(sys.argv[3])
    elif len(sys.argv) == 2: #single-use mode
        flank_idx = int(sys.argv[1])
        max_flank = None
        gauge_type = None

    else:
        print('')
        print('Script must be run with trailing index argument fo number of flanks:')
        print('e.g., 4_analyze_outputs.py 15')
        print('')
        sys.exit(0)
    op(path1, flank_idx, max_flank, gauge_type)