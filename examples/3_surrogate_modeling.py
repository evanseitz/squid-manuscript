# =============================================================================
# Perform surrogate modeling on ensemble of mutagenized genomic sequences..
# generated in the previous script
# =============================================================================
# Instructions: First, make sure that user inputs in 'set_parameters.py' are..
#               ..identical to those used in previous scripts during the current..
#               analysis. Also make sure to source the correct environment in the..
#               ..CLI corresponding to the chosen surrogate model (i.e., 'mavenn')
#               To note, since MAVE-NN requires Tensorflow 2.x, if the chosen..
#               deep learning model uses Tensorflow 1.x, it must be deactivated..
#               ('source deactivate') and replaced via 'source activate mavenn' 
#
#               Once the proper environment is sourced, the current script can..
#               ..be run for a single sequence or consecutive batch of sequences:
#
#               single:     Run: 'python 3_surrogate_modeling.py n' where 'n'..
#                           ..is the index (INT >= 0) of the desired sequence..
#                           ..matching an already-used index in the previous script 
#               batch:      In the corresponding '3_batch.sh' file, change..
#                           the range of sequences as desired
#                           (i.e., to match the range used in '2_batch.sh')
#                           To initialize the batch, run: 'bash 3_batch.sh' 
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings


def op(pyDir, df_idx):
    
    import os, sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings

    # =============================================================================
    # Import customized user parameters from script set_parameters.py
    # =============================================================================
    from set_parameters import set_params_1, set_params_2, set_params_3
      
    print("Importing model info, sequence data, and user parameters from set_parameters.py")

    GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
    comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(pyDir, False)
        
    if example == 'CAGI5-GOPHER' or example == 'CAGI5-ENFORMER':
        userDir = os.path.join(pyDir, 'examples_CAGI5')
    else:
        userDir = os.path.join(pyDir, 'examples_%s' % example)
    
    num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)
    
    surrogate, regression, gpmap, gauge, linearity, noise, noise_order, drop = set_params_3()

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
    
    if surrogate == 'mavenn':
        tf_ver = tf.__version__
        if tf_ver.startswith('1'):
            print('Tensorflow version: %s' % tf_ver)
            print('')
            print('MAVE-NN surrogate framework must be run with Tensorflow 2.x')
            print('')
            sys.exit(0)
        elif tf_ver.startswith('2'):
            import mavenn
            
    parentDir = os.path.dirname(pyDir)
    sys.path.append(pyDir)
    sys.path.append(parentDir)
    import squid.ink as squid_ink
    import squid.surrogate as squid_surrogate
    import squid.utils as squid_utils
    import squid.figs_mave as squid_figs_mave
    import squid.figs_surrogate as squid_figs_surrogate
    
    # =============================================================================
    # Rerun algorithms needed to match setup used in previous script
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
    squid_figs_surrogate.params_info(surrogate, regression, gpmap, gauge, linearity, noise, noise_order, drop, saveDir)

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
            
    # =============================================================================
    # Save figures of any attribution-based outputs generated in previous script
    # =============================================================================
    if compare is True:
        if 'ISM_single' in comparison_methods:
            ISM_logo = pd.read_csv(os.path.join(saveDir,'attributions_ISM_single.csv'), index_col=0)

            # sense correction
            '''if pred_transform == 'pca':
                ISM_attr = np.load(os.path.join(saveDir,'attributions_ISM_single.npy'))
                ISM_attr = ISM_attr[start:stop,:]
                TO BE DONE: for now, manually fix senses for ISM attribution maps in the respective 'sense' section of '4_analyze_outputs.py'
                '''

            if log2FC is True:
                ISM_logo= ISM_logo.fillna(0)
                ISM_logo.replace([np.inf, -np.inf], 0, inplace=True)

            if save is True:
                squid_figs_surrogate.single_logo(ISM_logo, 'ISM_single', False, start, stop, 3, saveDir)
        else:
            ISM_logo = None
            
        if 'ISM_double' in comparison_methods:
            ISM_double = np.load(os.path.join(saveDir,'attributions_ISM_double.npy'))
            if save is True:
                squid_figs_surrogate.ISM_double(ISM_double, start, stop, model_pad, alphabet, saveDir)
        else:
            ISM_double = None
            
        if 'saliency' in comparison_methods:
            sal_logo = pd.read_csv(os.path.join(saveDir,'attributions_saliency.csv'), index_col=0)
            if save is True:
                squid_figs_surrogate.single_logo(sal_logo, 'saliency', True, start, stop, 3, saveDir)
        else:
            sal_logo = None
            
        if 'deepLIFT' in comparison_methods or 'deepSHAP' in comparison_methods:
            dL_hypo_logo = pd.read_csv(os.path.join(saveDir,'attributions_deepLIFT_hypothetical.csv'), index_col=0)
            dL_contr_logo = pd.read_csv(os.path.join(saveDir,'attributions_deepLIFT_contribution.csv'), index_col=0)
            if save is True:
                squid_figs_surrogate.single_logo(dL_hypo_logo, 'deepLIFT_hypothetical', True, start, stop, 3, saveDir)
                squid_figs_surrogate.single_logo(dL_contr_logo, 'deepLIFT_contribution', False, start, stop, 3, saveDir)
        else:
            dL_hypo_logo = None
            dL_contr_logo = None
        

    # =============================================================================
    # Import MAVE dataset and set up for surrogate modeling
    # =============================================================================
    for file in os.listdir(saveDir):
        if file.startswith('mave_preds_unwrapped'):
            mave_path = os.path.join(saveDir, file)
    mave_custom = pd.read_pickle(mave_path, compression='gzip')
    
    # =============================================================================
    # Surrogate modeling and visualization
    # =============================================================================
    print('Initiating surrogate modeling...')
    start_time = time.time()
    if surrogate == 'ridge':
        coef, model, y_mave, yhat = squid_surrogate.run_ridge(mave_custom, squid_utils, alphabet=alphabet, drop=drop)

        print('--- Surrogate modeling: %s seconds ---' % (time.time() - start_time))

        coef = coef.reshape((len(mave_custom['x'][0]), len(alphabet)))

        logo_zeros = np.zeros((maxL,4))
        logo_zeros[start_full:stop_full,:] = coef
        logo = logo_zeros
        logo = squid_utils.arr2pd(logo, alphabet)
        
        squid_figs_surrogate.logo_additive(logo, scope, start, stop, 3, saveDir, 'ridge')
        squid_figs_surrogate.y_vs_yhat(model, y_mave, yhat, saveDir)
        logo.to_csv(os.path.join(saveDir, 'ridge_additive.csv'))
        np.save(os.path.join(saveDir,'ridge_coef.npy'), coef)
        import joblib
        joblib.dump(model, os.path.join(saveDir,'ridge_model.pkl'))


    elif surrogate == 'mavenn':
        if 0: #delimit dataframe here if the current (shorter) 'model_pad' was not applied previously in 2_generate_mave.py
            mave_custom['x'] = mave_custom['x'].str.slice(start_full,stop_full)

        mavenn_model, I_pred = squid_surrogate.run_mavenn(mave_custom, gpmap=gpmap, alphabet=alphabet,
                                                          gauge=gauge, regression=regression, linearity=linearity,
                                                          noise=noise, noise_order=noise_order, drop=drop)

        print('--- Surrogate modeling: %s seconds ---' % (time.time() - start_time))
        
        # fix gauge mode for model representation
        theta_dict = mavenn_model.get_theta(gauge=gauge) #for usage: theta_dict.keys()
        
        # embed (potentially-delimited) mavenn logo into max-length sequence
        mavenn_logo = theta_dict['logomaker_df']
        mavenn_logo.fillna(0, inplace=True) #if necessary, set NaN parameters to zero
        logo_zeros = np.zeros((maxL,4))
        logo_zeros[start_full:stop_full,:] = mavenn_logo
        mavenn_logo = logo_zeros
        # sense correction
        if pred_transform == 'pca':
            pred_scalar_wt = float(np.load(os.path.join(saveDir,'mave_scalar_WT.npy')))
            y_column = mave_custom['y']
            count_below = y_column[y_column < pred_scalar_wt].count()
            count_above = y_column[y_column >= pred_scalar_wt].count()
            if count_above > count_below:
                mavenn_logo *= -1.
        
        mavenn_logo = squid_utils.arr2pd(mavenn_logo, alphabet)
        
        if linearity == 'nonlinear':
            mavenn_model.save(os.path.join(saveDir,'mavenn_model'))
            mavenn_logo.to_csv(os.path.join(saveDir, 'logo_additive.csv'))
        elif linearity == 'linear':
            mavenn_model.save(os.path.join(saveDir,'mavenn_model_linear'))
            mavenn_logo.to_csv(os.path.join(saveDir, 'logo_additive_linear.csv'))
        if gpmap == 'pairwise':
            theta_lclc = theta_dict['theta_lclc']
            theta_lclc[np.isnan(theta_lclc)] = 0
            if linearity == 'nonlinear':
                np.save(os.path.join(saveDir,'mavenn_pairwise.npy'), theta_lclc)
            elif linearity == 'linear':
                np.save(os.path.join(saveDir,'mavenn_pairwise_linear.npy'), theta_lclc)
        trainval_df, test_df = mavenn.split_dataset(mave_custom)
            
        if save is True:
            squid_figs_surrogate.mavenn_info(mavenn_model, I_pred, saveDir) #save mavenn model information metrics to text
            squid_figs_surrogate.mavenn_performance(mavenn_model, I_pred, saveDir) #plot mavenn model performance
            squid_figs_surrogate.logo_additive(mavenn_logo, scope, start, stop, 3, saveDir, 'mavenn') #plot and save maveen additive logo
            squid_figs_surrogate.mavenn_yhat(mavenn_model, test_df, saveDir) #plot mavenn y versus yhat
            squid_figs_surrogate.mavenn_phi(mavenn_model, test_df, saveDir) #plot mavenn y versus phi
            if compare: #replot wideshot comparison of standard logos versus mavenn additive result
                halfwidth = 100 #expanded viewing window around motif center (full width is twice the halfwidth)            
                squid_figs_surrogate.compare_logos_ws(ISM_logo, sal_logo, mavenn_logo, scope, start, stop, model_pad, halfwidth, num_sim, maxL, map_crop, saveDir)
            if gpmap == 'pairwise':
                squid_figs_surrogate.mavenn_pairwise(theta_lclc, start, stop, model_pad, alphabet, saveDir)

    
    if clear_RAM is True: #potentially useful if looping over many sequences
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        del mave_custom
        
        
        
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
        print('Script must be run with trailing index argument: e.g., 3_surrogate_modeling.py 42')
        print('')
        sys.exit(0)
    op(path1, df_idx)