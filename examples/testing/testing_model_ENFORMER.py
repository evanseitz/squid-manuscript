# =============================================================================
# Script for testing predictions and wrapper functions for a deep learning model
# =============================================================================
# To use, first set necessary parameters for a desired deep learning model..
# ..in the set_parameters.py script; i.e., {example = 'CAGI5-ENFORMER'}. Then source..
# the proper environment (i.e., 'conda activate gopher' will work for Enformer)..
# and run via: python testing_model_ENFORMER.py
# =============================================================================

import os, sys, time
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
#import tensorflow.compat.v2 as tf
import tensorflow as tf
import matplotlib.pyplot as plt


pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2

GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(parentDir, True)

userDir = os.path.join(pyDir, 'examples_%s' % example)

num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)


method = 'agnostic' #options: cell-type {'agnostic' or 'matched'} summary statistic
assay = 'DNASE' #options: {'DNASE' or 'CAGE'}
table = pd.read_csv(os.path.join(parentDir, 'examples_ENFORMER/a_model_assets/suppl_table2.csv'))
assay_idx = table.index[table['assay_type']==assay].tolist()

maxL = 393216 #full input length
mut_pos = 196600 #location of example mutation (bp)
mut_bin_center = (mut_pos - ((maxL-114688)/2))/128. #see paper, re: 114688 bp
bin_center = 448 #actual bin center (i.e., 896/2)
OH = X_in[0] #e.g., index [0] for LDLR locus


if 1:
    #if 1: #pred from single OH
    pred_wt = get_prediction(np.expand_dims(OH, 0), example, model) #shape (1, 896, 5313)
    #else: #pred from array of OHs
        #pred_wt = get_prediction(X_in[0:10], example, model)
    print('Prediction:')
    print(pred_wt)
    if 1: #save prediction to file to expediate testing
        with h5py.File(os.path.join(pyDir,'enformer_pred_wt.h5'), 'w') as hf:
            hf.create_dataset('pred', data=pred_wt)

    # apply single mutation to WT
    X_in[0,mut_pos,0] = 0
    X_in[0,mut_pos,1] = 1
    OH_mut = X_in[0]
    pred_mut = get_prediction(np.expand_dims(OH_mut, 0), example, model)
    if 1: #save prediction to file to expediate testing
        with h5py.File(os.path.join(pyDir,'enformer_pred_mut.h5'), 'w') as hf:
            hf.create_dataset('pred', data=pred_mut)

else: #load in previously saved prediction
    with h5py.File(os.path.join(pyDir, 'enformer_pred_wt.h5'), 'r') as prediction:
        pred_wt = np.array(prediction['pred']).astype(np.float32)
    with h5py.File(os.path.join(pyDir, 'enformer_pred_mut.h5'), 'r') as prediction:
        pred_mut = np.array(prediction['pred']).astype(np.float32)


print('Unwrapped:')
pred_n = 0 #index of desired prediction (set to 0 if only one)
unwrap_wt = unwrap_prediction(pred_wt, class_idx, pred_n, example, pred_transform) #shape (896,)
print(unwrap_wt) #shape: (896,5313)
unwrap_mut = unwrap_prediction(pred_mut, class_idx, pred_n, example, pred_transform)

print('Compressed:')
compr_wt = compress_prediction(unwrap_wt, pred_transform='sum', pred_trans_delimit=5)
print(compr_wt)
compr_mut = compress_prediction(unwrap_mut, pred_transform='sum', pred_trans_delimit=5)
print(compr_mut)


if 1: #plot results
    plt.plot(pred_wt[0], color='k', alpha=.01)
    plt.plot(np.mean(pred_wt[0], axis=1), c='r', label='average', zorder=10)
    plt.axvline(bin_center, c='green')
    plt.legend()
    plt.title('All WT features: %s' % np.sum(np.mean(pred_wt[0], axis=1)))
    plt.tight_layout()
    plt.show()

    plt.plot(pred_wt[0][:,assay_idx], color='k', alpha=.01)
    plt.plot(unwrap_wt, c='r', label='average', zorder=10)
    plt.axvline(bin_center, c='green')
    plt.legend()
    plt.title('%s WT features: %s' % (assay, compr_wt))
    plt.tight_layout()
    plt.show()

    plt.plot(unwrap_wt, c='b', label='wt', zorder=10)
    plt.plot(unwrap_mut, c='r', label='mut', zorder=10)
    #plt.plot(np.mean(pred_wt[0][:,assay_idx], axis=1), c='b', label='wt', zorder=10)
    #plt.plot(np.mean(pred_mut[0][:,assay_idx], axis=1), c='r', label='mut', zorder=10)
    plt.axvline(bin_center, c='green')
    plt.legend()
    if assay == 'DNASE':
        plt.title('Fold change: %s' % (compr_mut - compr_wt))
    elif assay == 'CAGE':
        plt.title('Fold change: %s' % (np.log2(compr_mut+1) / np.log2(compr_wt+1)))
    plt.xlim(int(bin_center - 10), int(bin_center + 10))
    plt.tight_layout()
    plt.show()


if 0: #examine effects of bin window about mutation center on summary statistic
    print('Bin window summary statistics:')
    bin_in, bin_out = int(bin_center - 5), int(bin_center + 5)
    print('+-5:', np.sum(np.mean(pred_mut[0][:,assay_idx][bin_in:bin_out,:], axis=1) - np.mean(pred_wt[0][:,assay_idx][bin_in:bin_out,:], axis=1)))


if 1: #compute saliency scores for wildtype sequence
    print('Computing saliency scores...')
    gopherDir = os.path.join(parentDir, 'examples_GOPHER')
    sys.path.append(os.path.join(gopherDir,'a_model_assets/scripts'))
    import saliency_embed
    import tfomics
    import logomaker
    from matplotlib.ticker import MaxNLocator

    if example == 'CAGI5-ENFORMER':
        model_enformer = saliency_embed.Enformer(model)
        predictions = model_enformer.predict_on_batch(OH[np.newaxis])['human'][0] #shape (896, 5313)
        target_mask = np.zeros_like(predictions)
        for idx in np.arange(int((predictions.shape[0]/2)-pred_trans_delimit), int((predictions.shape[0]/2)+pred_trans_delimit)):
            target_mask[idx, class_idx] = 1
        saliency_scores = model_enformer.contribution_input_grad(OH.astype(np.float32), target_mask).numpy() #shape (393216, 4)
        #pooled_contribution_scores = tf.nn.avg_pool1d(np.abs(contribution_scores)[np.newaxis, :, np.newaxis], 128, 128, 'VALID')[0, :, 0].numpy()[1088:-1088]
        scores = np.expand_dims(saliency_scores, 0) #shape (1, 393216, 4)
        sal_logo = tfomics.impress.grad_times_input_to_df(np.expand_dims(OH, 0), scores)

    figpad = 150

    # plot saliency matrix heatmap
    fig, ax = plt.subplots(1,1, figsize=(50,5))
    import matplotlib
    norm = matplotlib.colors.TwoSlopeNorm(vmin=saliency_scores.min(), vcenter=0, vmax=saliency_scores.max())
    im = ax.pcolormesh(saliency_scores[int(maxL/2)-figpad:int(maxL/2)+figpad].T,
                linewidth=0,
                cmap='bwr',
                norm=norm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    B = ['A', 'C', 'G', 'T']
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(B, fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # plot saliency map (logo)
    fig, ax = plt.subplots(figsize=(10,2))
    logomaker.Logo(df=sal_logo[int(maxL/2)-figpad:int(maxL/2)+figpad],
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.tight_layout()
    plt.show()
