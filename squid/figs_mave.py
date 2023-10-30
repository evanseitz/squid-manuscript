import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import squid.utils as squid_utils


# save record of user parameters required for '2_generate_mave.py'
def params_info(motif_A, motif_B, motif_A_name, motif_B_name, max_muts, max_dist, rank_type,
                  comparison_methods, model_name, class_idx, alphabet, bin_res, output_skip, num_sim,
                  pred_transform, pred_trans_delimit, sort, use_mut, scope, model_pad, compare, map_crop,
                  saveDir):
    if os.path.exists(os.path.join(saveDir,'parameters_1.txt')):
        os.remove(os.path.join(saveDir,'parameters_1.txt'))
    f_out = open(os.path.join(saveDir,'parameters_1.txt'),'w')
    print('motif_A: %s' % (motif_A), file=f_out)
    print('motif_B: %s' % (motif_B), file=f_out)
    print('motif_A_name: %s' % (motif_A_name), file=f_out)
    print('motif_B_name: %s' % (motif_B_name), file=f_out)
    print('max_muts: %s' % (max_muts), file=f_out)
    print('max_dist: %s' % (max_dist), file=f_out)
    print('rank_type: %s' % (rank_type), file=f_out)
    print('comparison_methods: %s' % (comparison_methods), file=f_out)
    print('model_name: %s' % (model_name), file=f_out)
    print('class_idx: %s' % (class_idx), file=f_out)
    print('alphabet: %s' % (alphabet), file=f_out)
    print('bin_res: %s' % (bin_res), file=f_out)
    print('output_skip: %s' % (output_skip), file=f_out)
    print('num_sim: %s' % (num_sim), file=f_out)
    print('pred_transform: %s' % (pred_transform), file=f_out)
    print('pred_trans_delimit: %s' % (pred_trans_delimit), file=f_out)
    print('sort: %s' % (sort), file=f_out)
    print('use_mut: %s' % (use_mut), file=f_out)
    print('scope: %s' % (scope), file=f_out)
    print('model_pad: %s' % (model_pad), file=f_out)
    print('compare: %s' % (compare), file=f_out)
    print('map_crop: %s' % (map_crop), file=f_out)
    print('', file=f_out)
    f_out.close()
    

# plot deepnet prediction for WT sequence
def deepnet_pred_WT(scope, unwrap_pred, class_idx, start, stop, mut_range, bin_res, output_skip, saveDir):
    fig, ax = plt.subplots()
    ax.set_title('Pred WT')
    if bin_res > 1:
        ax.set_xlabel('position (%s-bin resolution)' % bin_res)
    else:
        ax.set_xlabel('position (base resolution)')
    ax.set_ylabel('model prediction')
    ax.plot(unwrap_pred)
    ax.axvline((start/float(bin_res))-output_skip, c='red', label='motif A', linewidth=1, zorder=10)
    if scope == 'inter':
        ax.axvline((stop/float(bin_res))-output_skip, c='green', label='motif B', linewidth=1, zorder=10)
    if scope != 'all':
        ax.axvline((mut_range[0]/float(bin_res))-output_skip, linestyle='--', c='gray', linewidth=1, zorder=10)
        ax.axvline((mut_range[1]/float(bin_res))-output_skip, linestyle='--', c='gray', linewidth=1, zorder=10)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'mave_pred_WT.pdf'), facecolor='w', dpi=200)
    plt.close()
    
    
# plot deepnet predictions for mutated sequences
def deepnet_pred_mut(mave, unwrap_pred, class_idx, num_sim, bin_res, saveDir):
    fig, ax = plt.subplots()
    for n in range(int(num_sim/10.)):
        ax.plot(mave['y'][n], c='k', alpha=0.01)
    ax.plot(unwrap_pred, c='red', linewidth=.5, zorder=10, label='WT')
    if bin_res > 1:
        ax.set_xlabel('position (%s-bin resolution)' % bin_res)
    else:
        ax.set_xlabel('position (base resolution)')
    ax.set_ylabel('model prediction')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, 'mave_predictions.pdf'), facecolor='w', dpi=200)
    plt.close()
    

# plot histogram of transformed deepnet predictions
def deepnet_y_hist(mave_custom, pred_scalar_wt, saveDir):
    fig, ax = plt.subplots()
    ax.hist(mave_custom['y'], bins=100)
    ax.set_xlabel('y')
    ax.set_ylabel('Frequency')
    ax.axvline(pred_scalar_wt, c='red', label='WT', linewidth=2, zorder=10)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'mave_distribution.pdf'), facecolor='w', dpi=200)
    plt.close()