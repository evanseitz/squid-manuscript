# =============================================================================
# Script for testing predictions and wrapper functions for a deep learning model
# =============================================================================
# To use, first set necessary parameters for a desired deep learning model..
# ..in the set_parameters.py script; i.e., {example = 'CAGI5-ENFORMER'}. Then source..
# the proper environment (i.e., 'conda activate gopher' will work for Enformer)..
# and run via: python testing_model_ENFORMER_mmap.py
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
bin_center = 448 #actual bin center (i.e., 896/2)


max_in_mem = 3 #total number of inputs allowed in one call to pred_to_batch(); based on system memory
num_preds = 7
one_hots = X_in[0:num_preds]
save_dir = pyDir
WT_pred_full = np.zeros(shape=(896,5313))
memmap_out = os.path.join(save_dir, 'memmap_preds_raw.npy')
if 1: #save batches of predictions to memmap
    pred_memmap = np.memmap(memmap_out, mode='w+', dtype='float32', 
                            shape=(num_preds, WT_pred_full.shape[0], WT_pred_full.shape[1]))

    def partition(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    batches = list(partition(np.arange(num_preds), max_in_mem))

    for batch_idx in range(len(batches)):
        print(batches[batch_idx])
        preds = get_prediction(one_hots[batches[batch_idx]], example, model)
        print(batch_idx, preds.shape)
        if len(batches[batch_idx]) > 1:
            print('IN:',preds[0,:,:])
        else:
            print('IN:',preds)

        pred_memmap[batches[batch_idx]] = preds

    #del pred_memmap

else: #load memap
    check = np.memmap(memmap_out, mode='r+', dtype='float32',
                      shape=(num_preds, WT_pred_full.shape[0], WT_pred_full.shape[1]))

    print('OUT 0:',check[0,:,:])
    print('OUT 3:',check[3,:,:])
    print('OUT 6:',check[6,:,:])




