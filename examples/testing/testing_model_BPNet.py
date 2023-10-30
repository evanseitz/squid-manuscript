# =============================================================================
# Script for testing predictions and wrapper functions for a deep learning model
# =============================================================================
# To use, first set necessary parameters for a desired deep learning model..
# ..in the set_parameters.py script; i.e., {example = 'BPNet'}. Then source..
# the proper environment (i.e., 'conda activate bpnet') and run via:
# python testing_model_BPNet.py
# =============================================================================


import os, sys
sys.dont_write_bytecode = True
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logomaker
pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandparentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandparentDir)
from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2
import squid.utils as squid_utils


GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_mut, max_dist, rank_type,\
comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(parentDir, True)
    
userDir = os.path.join(pyDir, 'examples_%s' % example)

num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)

#pred_transform = 'sum' #bypass for simplicity


# load sequence info for klf4 example
example_seq = 12269
start, stop = 546,546+len(motif_A)
TF, tf = 'Klf4', 'klf4'

OH = X_in[example_seq]
figpad = 3
if 1: #pred from single OH
    pred = get_prediction(np.expand_dims(OH, 0), example, model)
else: #pred from array of OHs
    pred = get_prediction(X_in[0:10], example, model)

print('Prediction:')
print(pred)



print('Unwrapped:')
pred_n = 0 #index of desired prediction (set to 0 if only one)
unwrap = unwrap_prediction(pred, class_idx, 0, example, pred_transform)
print(unwrap)


if 0:
    import keras.backend as K
    import tensorflow as tf
    #import keras.layers as kl
    #wn = kl.Lambda(lambda p: K.mean(K.sum(K.stop_gradient(tf.nn.softmax(p, dim=-2)) * p, axis=-2), axis=-1))(pred_tf) #https://github.com/kundajelab/bpnet-manuscript/blob/d7af1bda3ac8cc342b32f9cdac481ba55fe7ddca/basepair/heads.py#L322-L324
    #raw_outputs = pred[class_idx][:, :, 0]
    #wn = K.sum(K.stop_gradient(K.softmax(raw_outputs)) * raw_outputs, axis=-1) #https://github.com/kundajelab/bpnet-manuscript/blob/d7af1bda3ac8cc342b32f9cdac481ba55fe7ddca/basepair/BPNet.py#L78
    wn = K.sum(K.stop_gradient(K.softmax(unwrap)) * unwrap, axis=-1)
    #pred_tf = tf.convert_to_tensor(pred[class_idx])
    #wn = kl.Lambda(lambda p: K.mean(K.sum(K.stop_gradient(K.softmax(p, dim=-2)) * p, axis=-2), axis=-1))(pred_tf)
    with tf.Session() as sess:
        pred_wn = float(wn.eval())
    print('pred_wn', pred_wn)



print('Compressed:')
compr = compress_prediction(unwrap, pred_transform=pred_transform, 
                            pred_trans_delimit=pred_trans_delimit,
                            delimit_start=(start/float(bin_res))-output_skip, delimit_stop=(stop/float(bin_res))-output_skip)

print(compr)

if 0:
    import matplotlib.pyplot as plt
    plt.plot(pred[class_idx][pred_n][:,0])
    plt.plot(-1.*pred[class_idx][pred_n][:,1])
    plt.axhline(0, c='k', linewidth=1)
    plt.show()
    
if 1:  
    # =============================================================================
    # Generate attribution maps
    # =============================================================================
    # In Silico Mutagenesis (ISM)
    if 0:
        start_full = start - model_pad
        stop_full = start + len(motif_A) + model_pad
        if map_crop is False:
            ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                    unwrap_prediction, compress_prediction,
                                                    pred_transform, pred_trans_delimit, log2FC,
                                                    max_in_mem, None, None)
        elif map_crop is True:
            ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                                    unwrap_prediction, compress_prediction,
                                                    pred_transform, pred_trans_delimit, log2FC, 
                                                    max_in_mem, None, None, start=start_full, stop=stop_full)
        if 0:
            ISM_logo = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
        else:
            ISM_logo = squid_utils.arr2pd(ISM_df, alphabet=alphabet)


    from basepair.cli.imp_score import ImpScoreFile
    # load in imp_score.h5 (see https://github.com/kundajelab/bpnet-manuscript/bpnet-manuscript-data/output/..
        #../nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE/)
    isf = ImpScoreFile(os.path.join(parentDir,'examples_BPNet/a_model_assets/deeplift.imp_score.h5'),
                    default_imp_score='profile/wn')
    """
    ImpScoreFile :  isf.get_seq() #numpy array, shape: (147974, 1000, 4)
                    isf.get_contrib() #dict_keys(['Klf4', 'Nanog', 'Oct4', 'Sox2'])
                    isf.get_hyp_contrib() #dict_keys(['Klf4', 'Nanog', 'Oct4', 'Sox2'])
                    isf.get_profiles() #e.g., get_profiles(idx=0)
                        profiles = isf.get_profiles() 
                        print(profiles['Klf4'].shape)
                    isf.get_ranges() #pandas dataframe
                        #print(ranges.head())
                        #print(ranges['chrom'].value_counts())
    default_imp_score:
                    'profile/wn' #for profile (default)
                    'class/pre-act' #for binary
                    'weighted'
                    modisco_kwargs['grad_type']
                    modisco_kwargs['imp_scores']
    """

    ranges = isf.get_ranges()
    chr_idx = ranges.index[(ranges['chrom'] == 'chr1') | (ranges['chrom'] == 'chr8') | (ranges['chrom'] == 'chr9')].tolist()
    #chr8_idx = ranges.index[ranges['chrom'] == 'chr8'].tolist()
    #print(len(chr1_idx)) #len : 10219
    #print(len(chr8_idx)) #len : 9378
    #print(len(chr9_idx)) #len : 8130
    print(len(chr_idx)) #len : 27727

    if 0: #check chromosome info for sequence of interest
        seq = 13216 #Nanog; from folder 'rank27_seq13216'
        print('Sequence range:', ranges.iloc[chr_idx[seq]])
        seq = 1028 #Oct4; from folder 'rank27_seq13216'
        print('Sequence range:', ranges.iloc[chr_idx[seq]])
        seq = 22874 #Sox2; from folder 'rank27_seq13216'
        print('Sequence range:', ranges.iloc[chr_idx[seq]])
        #seqs = isf.get_seq()
        #print(seqs[seq,498:505,:])


    if 0: #save sequences from chromosomes 1,8,9 (i.e., the BPNet test set) to h5 file
        seqs = isf.get_seq()
        print(np.shape(seqs[chr_idx,:,:])) #shape : (27727, 1000, 4)
        
        f = h5py.File(os.path.join(pyDir,'bpnet_seqs_chr1-8-9.h5'), "w")
        dset = f.create_dataset("X", data=seqs[chr_idx,:,:], compression="gzip")
        dset = f.create_dataset("idx", data=chr_idx, compression="gzip")
        f.close()
        
        
    if 0: #save deeplift scores for chromosomes 1,8,9 to h5 file
        dL_hypo = isf.get_hyp_contrib(idx=chr_idx)
        #print(dL_hypo[TF].shape) #shape : (27727, 1000, 4)
        dL_contr = isf.get_contrib(idx=chr_idx)
        
        f = h5py.File(os.path.join(pyDir,'bpnet_deeplift_chr1-8-9.h5'), "w")
        dset = f.create_dataset("hypo_oct4", data=dL_hypo['Oct4'], compression="gzip")
        dset = f.create_dataset("hypo_sox2", data=dL_hypo['Sox2'], compression="gzip")
        dset = f.create_dataset("hypo_klf4", data=dL_hypo['Klf4'], compression="gzip")
        dset = f.create_dataset("hypo_nanog", data=dL_hypo['Nanog'], compression="gzip")
        dset = f.create_dataset("contr_oct4", data=dL_contr['Oct4'], compression="gzip")
        dset = f.create_dataset("contr_sox2", data=dL_contr['Sox2'], compression="gzip")
        dset = f.create_dataset("contr_klf4", data=dL_contr['Klf4'], compression="gzip")
        dset = f.create_dataset("contr_nanog", data=dL_contr['Nanog'], compression="gzip")
        f.close()
        
        dL_hypo_logo = squid_utils.arr2pd(dL_hypo[TF][example_seq], alphabet)
        dL_contr_logo = squid_utils.arr2pd(dL_contr[TF][example_seq], alphabet)

    else: #load deeplift scores, if already saved to file (above)
        with h5py.File(os.path.join(parentDir, 'examples_BPNet/a_model_assets/bpnet_deeplift_chr1-8-9.h5'), 'r') as dataset:
            dL_hypo = np.array(dataset['hypo_%s' % tf]).astype(np.float32)
            dL_contr = np.array(dataset['contr_%s' % tf]).astype(np.float32)
            
        dL_hypo_logo = squid_utils.arr2pd(dL_hypo[example_seq], alphabet)
        dL_contr_logo = squid_utils.arr2pd(dL_contr[example_seq], alphabet)


    # =============================================================================
    # Plot attribution maps
    # =============================================================================
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[10,2])
    axIdx = 0
    logomaker.Logo(df=ISM_logo[start-figpad:stop+figpad],
                    ax=axs[axIdx],
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    axs[axIdx].set_title('ISM', fontsize=14)
    axs[axIdx].xaxis.set_major_locator(MaxNLocator(integer=True))  
        
    logomaker.Logo(df=dL_hypo_logo[start-figpad:stop+figpad],
                    ax=axs[axIdx+1],
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    axs[axIdx+1].set_title('deepLIFT (hypothetical)', fontsize=14)
    axs[axIdx+1].xaxis.set_major_locator(MaxNLocator(integer=True))

    logomaker.Logo(df=dL_contr_logo[start-figpad:stop+figpad],
                    ax=axs[axIdx+2],
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    axs[axIdx+2].set_title('deepLIFT (contribution)', fontsize=14)
    axs[axIdx+2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()