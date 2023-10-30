import os, sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
import seaborn as sns
import random
random.seed(0)

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
import set_parameters
import squid.utils as squid_utils
from set_parameters import get_prediction, unwrap_prediction, compress_prediction

# environment: e.g., 'gopher'

# =============================================================================
# Load ResidualBind32 model and settings
# =============================================================================
alphabet = ['A','C','G','T']
alpha = 'dna'
bin_res = 32
output_skip = 0
class_idx = 13
userDir = os.path.join(parentDir, 'examples_GOPHER') #must already exist, with folder 'a_model_assets' propagated
sys.path.append(os.path.join(userDir,'a_model_assets/scripts'))
import utils
model, bin_size = utils.read_model(os.path.join(userDir,'a_model_assets/model_ResidualBind32_Exp_single'), compile_model = True)
with h5py.File(os.path.join(userDir, 'a_model_assets/cell_line_%s.h5' % class_idx), 'r') as dataset: #print(dataset.keys())
    testset = np.array(dataset['X']).astype(np.float32)
    targets = np.array(dataset['y']).astype(np.float32)


# =============================================================================
# Load sequence for SQUID_13_AP1_N_13_AP1_N_inter_mut0 : rank13_seq786_dist19
# =============================================================================
OH = testset[786]
start, stop = 1074, 1112+1
seq = squid_utils.oh2seq(OH, alphabet)
if 1: #pad=0
    start_A, stop_A = 1077, 1077+7
    start_C, stop_C = 1088, 1088+7
    start_B, stop_B = 1103, 1103+7
else: #pad=2
    start_A, stop_A = 1077-2, 1077+7+2
    start_C, stop_C = 1088-2, 1088+7+2
    start_B, stop_B = 1103-2, 1103+7+2

if 0:
    print('motif_region:',seq[start:stop])
    print('motif_A:',seq[start_A:stop_A])
    print('motif_C:',seq[start_C:stop_C])
    print('motif_B:',seq[start_B:stop_B])


# =============================================================================
# Perform motif occlusion experiment on wild-type sequence
# =============================================================================

def replacer(s, newstring, index, nofail=False):
    if not nofail and index not in range(len(s)): #raise an error if index is outside of the string
        raise ValueError("index outside given string")
    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring
    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + len(newstring):]


if 1: #wildtype occlusion plot
    from dinuc_shuffle import dinuc_shuffle

    n_background = 100
    df_1 = pd.DataFrame(columns = ['0', '1', '2', '3'])#, 'CTRL'])#, index=range(n_background*3))
    df_2 = pd.DataFrame(columns = ['WT', 'A', 'C', 'B', 'AC', 'BC', 'AB', 'ABC'], index=range(n_background))
    log2 = False

    pred = get_prediction(np.expand_dims(OH, 0), 'GOPHER', model)
    unwrap = unwrap_prediction(pred, class_idx, 0, 'GOPHER', pred_transform='sum')
    compr = compress_prediction(unwrap, pred_transform='sum', pred_trans_delimit=None)
    y_ABC = compr

    if log2 is True:
        y_ABC = np.log2(y_ABC)

    df_1.at[0, '0'] = y_ABC
    df_2.at[0, 'WT'] = y_ABC

    y_CTRL = []
    for z in range(n_background):
        print(z)
        random_A = ''.join(random.choices("ACGT", k=stop_A-start_A))
        random_C = ''.join(random.choices("ACGT", k=stop_C-start_C))
        random_B = ''.join(random.choices("ACGT", k=stop_B-start_B))
        ctrl = dinuc_shuffle(seq)

        seq_AB = replacer(seq, random_C, start_C)
        seq_AC = replacer(seq, random_B, start_B)
        seq_BC = replacer(seq, random_A, start_A)
        seq_A = replacer(seq_AB, random_B, start_B)
        seq_B = replacer(seq_BC, random_C, start_C)
        seq_C = replacer(seq_AC, random_A, start_A)
        seq_bg = replacer(seq_C, random_C, start_C)

        y_AB = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_AB, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_AC = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_AC, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_BC = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_BC, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_A = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_A, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_B = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_B, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_C = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_C, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_bg = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_bg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        y_ctrl = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(ctrl, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)

        if log2 is True:
            y_AB = np.log2(y_AB)
            y_AC = np.log2(y_AC)
            y_BC = np.log2(y_BC)
            y_A = np.log2(y_A)
            y_B = np.log2(y_B)
            y_C = np.log2(y_C)
            y_bg = np.log2(y_bg)
            y_ctrl = np.log2(y_ctrl)

        y_CTRL.append(y_ctrl)

        df_1 = df_1.append({'1':y_AB, '2':y_A, '3':y_bg}, ignore_index=True) #, 'CTRL':y_ctrl
        df_1 = df_1.append({'1':y_AC, '2':y_B}, ignore_index=True)
        df_1 = df_1.append({'1':y_BC, '2':y_C}, ignore_index=True)

        df_2.at[z, 'A'] = y_BC
        df_2.at[z, 'B'] = y_AC
        df_2.at[z, 'C'] = y_AB
        df_2.at[z, 'AB'] = y_C
        df_2.at[z, 'AC'] = y_B
        df_2.at[z, 'BC'] = y_A
        df_2.at[z, 'ABC'] = y_bg

    if 0: #sanity check
        print(seq[start:stop])
        print(seq_AB[start:stop])
        print(seq_AC[start:stop])
        print(seq_BC[start:stop])
        print(seq_A[start:stop])
        print(seq_B[start:stop])
        print(seq_C[start:stop])
        print(seq_bg[start:stop])

    print(df_1)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    df_1.plot(kind='box', ax=ax1)
    ax1.set_xlabel('Number of occluded AP-1 sites')
    if log2 is False:
        ax1.set_ylabel('DNN prediction')
    else:
        ax1.set_ylabel('log2 DNN prediction')
    ax1.axhline(np.median(np.array(y_CTRL)), color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir, 'occlusion.pdf'), facecolor='w', dpi=600)
        plt.close()
    else:
        plt.show()

    if 0:
        print(df_2)
        fig, ax2 = plt.subplots(figsize=(6, 6))
        df_2.plot(kind='box', ax=ax2)
        ax2.set_xlabel('Occluded AP-1 sites')
        if log2 is False:
            ax2.set_ylabel('DNN prediction')
        else:
            ax2.set_ylabel('log2 DNN prediction')
        plt.tight_layout()
        plt.show()


if 0: #wildtype occlusion fold-change plot
    n_background = 100
    df = pd.DataFrame(columns = ['bg', 'A', 'B', 'C', 'AC', 'AB', 'BC', 'ABC'], index=range(n_background))

    pred = get_prediction(np.expand_dims(OH, 0), 'GOPHER', model)
    unwrap = unwrap_prediction(pred, class_idx, 0, 'GOPHER', pred_transform='sum')
    compr = compress_prediction(unwrap, pred_transform='sum', pred_trans_delimit=None)
    df['ABC'] = compr

    for z in range(n_background):
        print(z)
        random_A = ''.join(random.choices("ACGT", k=stop_A-start_A))
        random_B = ''.join(random.choices("ACGT", k=stop_B-start_B))
        random_C = ''.join(random.choices("ACGT", k=stop_C-start_C))

        seq_AB = replacer(seq, random_C, start_C)
        seq_AC = replacer(seq, random_B, start_B)
        seq_BC = replacer(seq, random_A, start_A)
        seq_A = replacer(seq_AB, random_B, start_B)
        seq_B = replacer(seq_BC, random_C, start_C)
        seq_C = replacer(seq_AC, random_A, start_A)
        seq_bg = replacer(seq_C, random_C, start_C)

        df.at[z, 'AB'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_AB, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'AC'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_AC, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'BC'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_BC, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'A'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_A, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'B'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_B, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'C'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_C, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
        df.at[z, 'bg'] = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_bg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)

    if 0: #sanity check
        print(seq[start:stop])
        print(seq_AB[start:stop])
        print(seq_AC[start:stop])
        print(seq_BC[start:stop])
        print(seq_A[start:stop])
        print(seq_B[start:stop])
        print(seq_C[start:stop])
        print(seq_bg[start:stop])

    print(df)
    df_fc = pd.DataFrame(columns = ['AB', 'AC', 'BC'], dtype=float, index=range(n_background))
    if 0: #DeepSTARR-esque definition of cooperativity
        df_fc['AB'] =  df['AB'] / (df['A'] + df['B'] - df['bg'])
        df_fc['AC'] =  df['AC'] / (df['A'] + df['C'] - df['bg'])
        df_fc['BC'] =  df['BC'] / (df['B'] + df['C'] - df['bg'])
        #df_fc['ABC'] = df['ABC'] / (df['AB'] + df['AC'] + df['BC'] - 5*df['bg'])
    elif 1: #classic definition (v1)
        df_fc['AB'] =  df['AB'] -  df['A'] -  df['B'] +  df['bg']
        df_fc['AC'] =  df['AC'] -  df['A'] -  df['C'] +  df['bg']
        df_fc['BC'] =  df['BC'] -  df['B'] -  df['C'] +  df['bg']
    elif 0: #classic definition (v2)
        df_fc['AB'] =  df['AB']* df['bg'] / (df['A'] * df['B'])
        df_fc['AC'] =  df['AC']* df['bg'] / (df['A'] * df['C'])
        df_fc['BC'] =  df['BC']* df['bg'] / (df['B'] * df['C'])

    print(df_fc)

    fig, ax = plt.subplots(figsize=(6, 6))
    df_fc.plot(kind='box', ax=ax)
    ax.axhline(y=1, c='gray', linestyle='dashed')
    ax.set_ylabel('Cooperativity')
    plt.tight_layout()
    plt.show()

if 0: #GIA distance experiment with random backgrounds
    n_background = 100
    dists = np.arange(0,20)
    df_dists = pd.DataFrame(columns = ['dist','gia'], index=range(n_background*len(dists)))
    motif_1 = 'TGATTCA'
    start_1 = start_A
    motif_2 = 'CGAATCA'
    start_2 = start_B

    dz_idx = 0
    for d in dists:
        for z in range(n_background):
            print(d, z)
            df_dists.at[dz_idx, 'dist'] = d
            seq_bg = ''.join(random.choices("ACGT", k=2048))
            seq_bg = replacer(seq_bg, motif_1, start_1)
            y_bg = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_bg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
            seq_fg = replacer(seq_bg, motif_2, start_1+len(motif_1)+d)
            y_fg = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_fg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
            df_dists.at[dz_idx, 'gia'] = y_fg - y_bg
            dz_idx += 1

        if 0: #sanity check
            print(seq_fg[start_1:start_1+len(motif_1)+d+len(motif_2)])

    print(df_dists.head())
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=df_dists, x='dist', y='gia')
    plt.tight_layout()
    plt.show()

if 0: #GIA distance experiment with exact wildtype binding sites
    n_background = 1000
    dists = np.arange(2,20)
    df_dists = pd.DataFrame(columns = ['dist','gia'], index=range(n_background*len(dists)))
    if 1:
        wildtype_A = 'CCTGATTCAGA'
        wildtype_C = 'AGCGAATCACC'
        wildtype_B = 'GATGATTCAAC'
        pad = 2
    else:
        wildtype_A = 'TGATTCA'
        wildtype_C = 'CGAATCA'
        wildtype_B = 'TGATTCA'
        pad = 0

    dz_idx = 0
    for d in [4-pad, 8-pad, 19-pad]:#dists:
        if d == 4-pad:
            motif_1 = wildtype_A
            start_1 = start_A - pad
            motif_2 = wildtype_C
            start_2 = start_C - pad
        if d == 8-pad:
            motif_1 = wildtype_C
            start_1 = start_C - pad
            motif_2 = wildtype_B
            start_2 = start_B - pad
        if d == 19-pad:
            motif_1 = wildtype_A
            start_1 = start_A - pad
            motif_2 = wildtype_B
            start_2 = start_B - pad
        print(d+pad, start_1, start_2)

        for z in range(n_background):
            #print(d, z)
            df_dists.at[dz_idx, 'dist'] = d+pad
            seq_bg = ''.join(random.choices("ACGT", k=2048))
            seq_bg = replacer(seq_bg, motif_1, start_1)
            y_bg = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_bg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
            seq_fg = replacer(seq_bg, motif_2, start_1+len(motif_1)+d-pad)
            y_fg = compress_prediction(unwrap_prediction(get_prediction(np.expand_dims(squid_utils.seq2oh(seq_fg, alphabet), 0), 'GOPHER', model), class_idx, 0, 'GOPHER', pred_transform='sum'), pred_transform='sum', pred_trans_delimit=None)
            df_dists.at[dz_idx, 'gia'] = y_fg - y_bg
            dz_idx += 1

        if 0: #sanity check
            print(seq_fg[start_1-5:start_1+len(motif_1)+d-pad+len(motif_2)+5])

    print(df_dists.head())
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=df_dists, x='dist', y='gia')
    plt.tight_layout()
    plt.show()