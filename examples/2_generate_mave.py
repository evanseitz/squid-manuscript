# =============================================================================
# Generate an ensemble of mutagenized genomic sequences in conjunction with..
# ..respective predictions from a user-defined deep learning model to form a..
# ..MAVE dataset, which is used to perform surrogate modeling in the next script
# =============================================================================
# Instructions: First, make sure to adjust user inputs in 'set_parameters.py'..
#               ..as desired, following the documentation there. Before running..
#               ..this script, make sure to source the correct environment..
#               ..in the CLI corresponding to the chosen deep learning model
#
#               Once an environment is sourced, the current script can be run..
#               ..for a single sequence or consecutive batch of sequences:
#
#               single:     Run: 'python 2_generate_mave.py n' where 'n' is..
#                           ..the index (INT >= 0) of the desired sequence
#               batch:      In the corresponding '2_batch.sh' file, change..
#                           the range of sequences as desired.
#                           To initialize the batch, run: 'bash 2_batch.sh' 
#
#               For our example pipeline, the sequences used for surrogate..
#               ..modeling are defined by the motif-instance dataframes provided..
#               ..in the 'b_recognition_sites' folder. These can also be sorted..
#               ..in the 'if sort is True' section below
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings


def op(pyDir, df_idx):
    
    import os, sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings
    parentDir = os.path.dirname(pyDir)
    sys.path.append(pyDir)
    sys.path.append(parentDir)
    import squid.ink as squid_ink
    import squid.utils as squid_utils
    import squid.figs_mave as squid_figs_mave

    # =============================================================================
    # Import customized user parameters from script set_parameters.py
    # =============================================================================
    from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2
    
    print("Importing model info, sequence data, and user parameters from set_parameters.py")

    GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
    comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(pyDir, True)
        
    if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
        userDir = os.path.join(pyDir, 'examples_CAGI5')
    else:
        userDir = os.path.join(pyDir, 'examples_%s' % example)
    
    num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)

    if example == 'TFChIP': #override
        sort = False
        compare = False

    # =============================================================================
    # Import dependencies
    # =============================================================================
    print('Importing dependencies...')
    
    sys.dont_write_bytecode = True
    import time
    import gc
    import numpy as np
    import pickle
    import pandas as pd
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    import tensorflow as tf
    # turn off tensorflow warnings:
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    import tfomics
    
    if compare is True:
        if 'saliency' in comparison_methods:
            sys.path.append(os.path.join(os.path.join(pyDir, 'examples_GOPHER'),'a_model_assets/scripts')) #ZULU, needs to take in function wrapper
            import saliency_embed

    # =============================================================================
    # Run algorithm based on deep learning model and user-defined settings
    # =============================================================================
    # import ordered dataframe of highest-scoring motifs
    if scope == 'intra':
        if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER' or example == 'TFChIP':
            motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s_positions.csv' % (motif_A_name)))
        else:
            motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s/%s_positions.csv' % (model_name, motif_A_name)))
    elif scope == 'inter':
        if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
            print('Using default pipeline for CAGI5...')
            motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s_positions.csv' % (motif_A_name)))
        else:
            motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s/%s_%s_positions.csv' % (model_name, motif_A_name, motif_B_name)))

    if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
        if model_pad == None:
            model_pad = int(np.ceil(int(motif_info.iloc[df_idx]['locus_len'])/2.))
            print('')
            print('model_pad updated:', model_pad)

    maxL = X_in.shape[1] #maximum length of sequence
    motif_A_len = len(motif_A)
    if scope == 'inter':
        motif_B_len = len(motif_B)
        
    if scope == 'intra':
        saveDir = os.path.join(userDir,'c_surrogate_outputs/%s/SQUID_%s_%s_mut%s' % (model_name, motif_A_name, scope, use_mut)) #location of output directory
    elif scope == 'inter':
        saveDir = os.path.join(userDir,'c_surrogate_outputs/%s/SQUID_%s_%s_%s_mut%s' % (model_name, motif_A_name, motif_B_name, scope, use_mut)) #location of output directory
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        
    # save user parameters to text:
    squid_figs_mave.params_info(motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist,
                                rank_type, comparison_methods, model_name, class_idx, alphabet, bin_res, output_skip,
                                num_sim, pred_transform, pred_trans_delimit, sort, use_mut, scope, model_pad, compare, map_crop,
                                saveDir)
    
    # =============================================================================
    # Sort dataframe based on a given condition (customize as desired)
    # Sorting method must remain the same between all scripts for a given analysis
    # =============================================================================
    if sort is True:        
        if scope == 'intra':
            motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
            motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]
        elif scope == 'inter':
            motif_info = motif_info.sort_values(by = ['motif_rank','inter_dist'], ascending = [False, True])
            motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]
            
        if example == 'BPNet':
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 400].index)
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 600].index)
        if example == 'DeepSTARR':
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] < 80].index) #110 for gata, 80 for creb
            motif_info = motif_info.drop(motif_info[motif_info['motif_start'] > 170].index) #140 for gata, 170 for creb

        motif_info.reset_index(drop=True,inplace=True)

    # =============================================================================
    # Collect information pertaining to the current input sequence
    # =============================================================================
    print('')
    if scope == 'intra':
        seq_idx = motif_info.loc[df_idx][0]
        start = motif_info.loc[df_idx][4]
        stop = start + motif_A_len
    elif scope == 'inter':
        seq_idx = motif_info.loc[df_idx][0]
        start = motif_info.loc[df_idx][4]
        dist = motif_info.loc[df_idx][5]
        stop = start + motif_A_len + dist + motif_B_len
    print(motif_info.loc[df_idx])
        
    if model_pad != 'full':
        start_full = start - model_pad
        stop_full = stop + model_pad
    else:
        start_full = 0
        stop_full = maxL

    print('')
    if (start_full < 0) or (stop_full) > maxL:
        print('Encountered error while padding due to extension beyond sequence boundary...')
        print('Resetting padding up to boundary edge')
        if start_full < 0:
            start_full = 0
        if stop_full > maxL:
            stop_full = maxL
            
    if scope == 'inter':
        saveDir = saveDir + '/rank%s_seq%s_dist%s' % (df_idx, seq_idx, dist)
    else:
        saveDir = saveDir + '/rank%s_seq%s' % (df_idx, seq_idx)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    OH = X_in[seq_idx]
    pred_all_wt = get_prediction(np.expand_dims(OH, 0), example, model)
    
    # =============================================================================
    # Gold-standard first-order attribution methods
    # =============================================================================
    if compare is True:
        if 'ISM_single' in comparison_methods:
            print('Calculating single ISM attribution map...')

            start_time = time.time()
            if map_crop is False:
                ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                        unwrap_prediction, compress_prediction,
                                                        pred_transform, pred_trans_delimit, log2FC,
                                                        max_in_mem, save, saveDir,
                                                        delimit_start=(start/float(bin_res))-output_skip, delimit_stop=(stop/float(bin_res))-output_skip)
            elif map_crop is True:
                ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                        unwrap_prediction, compress_prediction,
                                                        pred_transform, pred_trans_delimit, log2FC,
                                                        max_in_mem, save, saveDir,
                                                        start=start_full, stop=stop_full,
                                                        delimit_start=(start/float(bin_res))-output_skip, delimit_stop=(stop/float(bin_res))-output_skip)
                
            np.save(os.path.join(saveDir,'attributions_ISM_single.npy'), ISM_df)
            ISM_logo = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)

            print("Single ISM: --- %s seconds ---" % (time.time() - start_time))
        else:
            ISM_logo = None
            
                
        if 'saliency' in comparison_methods:
            start_time = time.time()
            print('Calculating saliency attribution map...')
            if example != 'CAGI5-ENFORMER':
                explainer = saliency_embed.Explainer(model, example, class_index=class_idx) #tfomics.explain.Explainer(model, class_index=class_idx)
                saliency_scores = explainer.saliency_maps(np.expand_dims(OH, 0))
                np.save(os.path.join(saveDir,'attributions_saliency.npy'), saliency_scores[0])
                scores = np.expand_dims(saliency_scores[0], axis=0)
            else:
                model_enformer = saliency_embed.Enformer(model)
                predictions = model_enformer.predict_on_batch(OH[np.newaxis])['human'][0] #shape (896, 5313)
                target_mask = np.zeros_like(predictions)
                #for idx in np.arange(int((predictions.shape[0]/2)-pred_trans_delimit), int((predictions.shape[0]/2)+pred_trans_delimit)):
                for idx in np.arange(((start/float(bin_res))-output_skip)-pred_trans_delimit, ((stop/float(bin_res))-output_skip)+pred_trans_delimit):
                    target_mask[idx, class_idx] = 1
                saliency_scores = model_enformer.contribution_input_grad(OH.astype(np.float32), target_mask).numpy() #shape (393216, 4)
                np.save(os.path.join(saveDir,'attributions_saliency.npy'), saliency_scores)
                scores = np.expand_dims(saliency_scores, 0) #shape (1, 393216, 4)
            sal_logo = tfomics.impress.grad_times_input_to_df(np.expand_dims(OH, 0), scores)
            print("Saliency map: --- %s seconds ---" % (time.time() - start_time))
        else:
            sal_logo = None
            
            
        if 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
            start_time = time.time()
            if example == 'DeepSTARR':
                print('Calculating deepSHAP attribution map...')
                from deepstarr_deepshap import deepExplainer
                dL_df = deepExplainer(model, np.expand_dims(OH, 0), class_output=class_idx)
                dL_hypo = dL_df[0]
                dL_contr = dL_df[1]
                dL_hypo_logo = squid_utils.arr2pd(dL_hypo[0], alphabet)
                dL_contr_logo = squid_utils.arr2pd(dL_contr[0], alphabet)
                print("deepSHAP: --- %s seconds ---" % (time.time() - start_time))
            if example == 'BPNet':
                print('Calculating deepLIFT attribution map...')
                import h5py
                h5_file = os.path.join(userDir, 'a_model_assets/bpnet_deeplift_chr1-8-9.h5')
                # see 'SQUID/examples/testing/testing_model_BPNet.py' for generating the above subset of deeplift scores..
                # originally from trained model: https://github.com/kundajelab/bpnet-manuscript/bpnet-manuscript-data/output/..
                # ../nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE/
                if os.path.exists(h5_file):
                    with h5py.File(h5_file, 'r') as dataset:
                        if class_idx == 'Oct4/profile':
                            dL_hypo = np.array(dataset['hypo_oct4']).astype(np.float32)
                            dL_contr = np.array(dataset['contr_oct4']).astype(np.float32)
                        elif class_idx == 'Sox2/profile':
                            dL_hypo = np.array(dataset['hypo_sox2']).astype(np.float32)
                            dL_contr = np.array(dataset['contr_sox2']).astype(np.float32)
                        elif class_idx == 'Klf4/profile':
                            dL_hypo = np.array(dataset['hypo_klf4']).astype(np.float32)
                            dL_contr = np.array(dataset['contr_klf4']).astype(np.float32)
                        elif class_idx == 'Nanog/profile':
                            dL_hypo = np.array(dataset['hypo_nanog']).astype(np.float32)
                            dL_contr = np.array(dataset['contr_nanog']).astype(np.float32)
                        
                    dL_hypo_logo = squid_utils.arr2pd(dL_hypo[seq_idx], alphabet)
                    dL_contr_logo = squid_utils.arr2pd(dL_contr[seq_idx], alphabet)
                else:
                    print('DeepLIFT data for test set not found...')
                    print('Generate using script /testing/testing_model_BPNet.py')
                    dL_hypo_logo = None
                    dL_contr_logo = None
                print("deepLIFT: --- %s seconds ---" % (time.time() - start_time))
                
        else:
            dL_hypo_logo = None
            dL_contr_logo = None
            
    
        if ISM_logo is not None:
            ISM_logo.to_csv(os.path.join(saveDir,'attributions_ISM_single.csv'))
        if sal_logo is not None:
            sal_logo.to_csv(os.path.join(saveDir,'attributions_saliency.csv'))
        if dL_hypo_logo is not None and dL_contr_logo is not None:
            dL_hypo_logo.to_csv(os.path.join(saveDir,'attributions_deepLIFT_hypothetical.csv'))
            dL_contr_logo.to_csv(os.path.join(saveDir,'attributions_deepLIFT_contribution.csv'))
            
 
    # =============================================================================
    # Second-order ("double") ISM: not recommended for large sequence lengths due to memory constraints
    # =============================================================================
    if compare is True and 'ISM_double' in comparison_methods: 
        start_time = time.time()
        print('Calculating double ISM attribution map...')
        ISM_matrix = squid_utils.ISM_double(OH, model=model, start=start_full, stop=stop_full, model_type=userDir, class_num=class_idx)
        print("--- Double ISM: %s seconds ---" % (time.time() - start_time))  
        np.save(os.path.join(saveDir,'attributions_ISM_double.npy'), ISM_matrix)
    
    # =============================================================================
    # Generate in silico MAVE using deep learning model predictions
    # =============================================================================
    print('Preparing MAVE...')
    # due to memory constraints, must delimit region to perform mutagenesis on sequence for pairwise analysis
    mut_range = [start_full, stop_full] #range of nucleotides to mutate on input sequence
    
    # ad hoc – determined heuristically for GOPHER model:
    if int(stop_full-start_full) <= 200:
        avg_num_mut = int(np.ceil((stop_full-start_full)/10.))
    else:
        avg_num_mut = 20 #ad hoc for now
    
    start_time = time.time()
    dS = squid_ink.deep_sea(OH, num_sim=num_sim, avg_num_mut=avg_num_mut, alphabet=''.join(alphabet), mode='poisson', mut_range=mut_range)
    print('--- Deep sequencing: %s seconds ---' % (time.time() - start_time))
        
    unwrap_pred_wt = unwrap_prediction(pred_all_wt, class_idx, 0, example, pred_transform)

    if save is True and bin_res is not None: #plot deepnet prediction for WT sequence
        squid_figs_mave.deepnet_pred_WT(scope, unwrap_pred_wt, class_idx, start, stop, mut_range, bin_res, output_skip, saveDir)
        
    start_time = time.time()
    mave, pred_wt = squid_ink.calamari(dS, OH, model, class_idx, alphabet, GPU, example, get_prediction, unwrap_prediction, compress_prediction, unwrap_pred_wt,
                              pred_transform, pred_trans_delimit, start, stop, bin_res, output_skip, max_in_mem, saveDir, squid_utils, mut_range=mut_range)

    print('--- Generating MAVE dataset: %s seconds ---' % (time.time() - start_time))

    #if save is True and bin_res is not None: #plot and save deepnet predictions for mutated sequences
        #squid_figs_mave.deepnet_pred_mut(mave, unwrap_pred_wt, class_idx, num_sim, bin_res, saveDir)


    mave_custom = mave.copy()
    print(mave_custom)

    
    if log2FC is True:
        pred_min = mave_custom['y'].min()
        mave_custom['y'] += (abs(pred_min) + 1)
        pred_wt += (abs(pred_min) + 1)
        pred_wt = np.log2(pred_wt)
        mave_custom['y'] = mave_custom['y'].apply(lambda x: np.log2(x))
        mave_custom['y'] -= pred_wt
        pred_wt = 0
        #mave_custom['y'] = mave_custom['y'].fillna(0)
        #mave_custom.replace([np.inf, -np.inf], 0, inplace=True)
        
    # ensure proper format for keras/TF
    mave_custom['y'] = mave_custom['y'].apply(lambda x: np.asarray(x).astype('float32'))
    
    # mandatory save for subsequent script
    mave_custom.to_pickle(os.path.join(saveDir,'./mave_preds_unwrapped.csv.gz'), compression='gzip')
    if pred_transform == 'pca':
        np.save(os.path.join(saveDir,'mave_scalar_WT.npy'), pred_wt)
    
    if save is True: #plot histogram of final deepnet predictions
        squid_figs_mave.deepnet_y_hist(mave_custom, pred_wt, saveDir)
        
        
    if clear_RAM is True:
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

 
if __name__ == '__main__':
    #path0 = os.path.splitext(sys.argv[0])[0]
    #path1, tail = os.path.split(path0)
    path1 = os.path.dirname(os.path.abspath(__file__))

    """
    df_idx : INT >= 0
        Index of a sequence where surrogate modeling is to be applied
        If following our example pipeline, each 'df_idx' corresponds..
        ..to a unique instance of a recognition site as discovered..
        ..in the previous script '1_locate_patterns.py'
        (e.g., see 'AP1_positions.csv' and 'AP1_AP1_positions.csv')
    """
    
    if len(sys.argv) > 1:
        df_idx = int(sys.argv[1])

    else:
        print('')
        print('Script must be run with trailing index argument: e.g., 2_generate_mave.py 42')
        print('')
        sys.exit(0)
    op(path1, df_idx)