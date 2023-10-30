# =============================================================================
# Script for testing predictions and wrapper functions for a deep learning model
# =============================================================================
# To use, first set necessary parameters for a desired deep learning model..
# ..in the set_parameters.py script; i.e., {example = 'DeepSTARR'}. Then source..
# the proper environment (i.e., 'conda activate deepstarr') and run via:
# python testing_model_DeepSTARR.py
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

GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,\
comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(parentDir, True)

num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)

# load sequence info for GATA example
figpad = 3
if 0: #DRE
    OH = X_in[1134]
    start, stop = 159, 159+8
    class_idx = 1
    #print(squid_utils.oh2seq(OH, ['A','C','G','T']))
elif 0: #GATA
    OH = X_in[24869]
    start, stop = 123, 123+5
    class_idx = 0
elif 0: #AP1
    OH = X_in[13748]
    start, stop = 120, 120+7
    class_idx = 0
    print(squid_utils.oh2seq(OH[start:stop], ['A','C','G','T']))
    #print(squid_utils.oh2seq(OH, ['A','C','G','T']))
elif 1: #Ohler1
    OH = X_in[22627]
    start, stop = 119, 119+9
    class_idx = 1
    #print(squid_utils.oh2seq(OH[start:stop], ['A','C','G','T']))
    print(squid_utils.oh2seq(OH, ['A','C','G','T']))

if 1: #pred from single OH
    pred = get_prediction(np.expand_dims(OH, 0), example, model)
else: #pred from array of OHs
    pred = get_prediction(X_in[0:10], example, model)
    
print('Prediction:')
print(pred)
print('Unwrapped:')
pred_n = 0 #index of desired prediction (set to 0 if only one)
unwrap = unwrap_prediction(pred, class_idx, pred_n, example, pred_transform)
print(unwrap)
print('Compressed:')
compr = compress_prediction(unwrap, pred_transform=None, pred_trans_delimit=pred_trans_delimit)
print(compr)


# =============================================================================
# Generate attribution maps
# =============================================================================
if 1:
    # In Silico Mutagenesis (ISM)
    ISM_df = squid_utils.ISM_single(OH, model, class_idx, example, get_prediction,
                                    unwrap_prediction, compress_prediction, pred_transform, 
                                    pred_trans_delimit, log2FC, max_in_mem, None, None,
                                    start=start-figpad, stop=stop+figpad)

    #ISM_logo = squid_utils.l2_norm_to_df(OH, ISM_df, alphabet=alphabet, alpha=alpha)
    ISM_logo = squid_utils.arr2pd(ISM_df, alphabet=alphabet)

    # deepLIFT
    from deepstarr_deepshap import deepExplainer

    dL_df = deepExplainer(model, np.expand_dims(OH, 0), class_output=class_idx)

    dL_hypo = dL_df[0]
    dL_contr = dL_df[1]

    dL_hypo_logo = squid_utils.arr2pd(dL_hypo[0], alphabet)
    dL_contr_logo = squid_utils.arr2pd(dL_contr[0], alphabet)


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
                    center_values=False,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    axs[axIdx+1].set_title('deepLIFT (hypothetical)', fontsize=14)
    axs[axIdx+1].xaxis.set_major_locator(MaxNLocator(integer=True))

    logomaker.Logo(df=dL_contr_logo[start-figpad:stop+figpad],
                    ax=axs[axIdx+2],
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=False,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    axs[axIdx+2].set_title('deepLIFT (contribution)', fontsize=14)
    axs[axIdx+2].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()