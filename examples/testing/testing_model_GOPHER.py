# =============================================================================
# Script for testing predictions and wrapper functions for a deep learning model
# =============================================================================
# To use, first set necessary parameters for a desired deep learning model..
# ..in the set_parameters.py script; i.e., {example = 'GOPHER'}. Then source..
# the proper environment (i.e., 'conda activate gopher') and run via:
# python testing_model_GOPHER.py
# =============================================================================


import os, sys
sys.dont_write_bytecode = True
import numpy as np
import shutil
import zipfile
import h5py

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils
from set_parameters import get_prediction, unwrap_prediction, compress_prediction, set_params_1, set_params_2

GPU, example, motif_A, motif_B, motif_A_name, motif_B_name, max_mut, max_dist, rank_type,\
comparison_methods, model_name, class_idx, log2FC, alphabet, alpha, bin_res, output_skip, model, X_in = set_params_1(parentDir, True)
    
num_sim, pred_transform, pred_trans_delimit, scope, sort, use_mut, model_pad, compare, map_crop, max_in_mem, clear_RAM, save = set_params_2(example)

userDir = os.path.join(pyDir, 'examples_%s' % example)

OH = X_in[0]
if 0: #check WT sequence
    seq = squid_utils.oh2seq(OH, ['A','C','G','T'])
    print(seq[1190:1230])


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
compr = compress_prediction(unwrap, pred_transform='sum', pred_trans_delimit=pred_trans_delimit)
print(compr)


if 1:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    color=iter(cm.tab20(np.linspace(0, 1, 15)))
    plt.title('All Classes')
    for i in range(15):
        c=next(color)
        if i == class_idx:
            ci = c
        plt.plot(pred[0,:,i], color=c, label='%s' % i)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #import statistics
    #temp_stats = np.argmax(pred[0,:,:], axis=-1)
    #mode = statistics.mode(temp_stats)
    avgs = np.mean(pred[0,:,:], axis=0)
    print('Most active class:', np.argmax(avgs))

    #plt.title('Class %s' % class_idx)
    #plt.plot(pred[0,:,class_idx], color=ci)
    #plt.tight_layout()
    #plt.show()

    for i in range(15):
        plt.plot(pred[0,:,i], color='k', alpha=.1)
    plt.plot(np.mean(pred[0,:,:], axis=1), c='k', label='average')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if 1: #redo for getting average of predictions {example = 'CAGI5'}
        pred = get_prediction(np.expand_dims(OH, 0), 'CAGI5-GOPHER', model)
        unwrap = unwrap_prediction(pred, class_idx, pred_n, 'CAGI5-GOPHER', pred_transform)
        plt.plot(unwrap, c='k')
        plt.tight_layout()
        plt.show()
        #compr = compress_prediction(unwrap, pred_transform='sum')