# =============================================================================
# Preprocess Critical Assessment of Genome Interpretation 5 (CAGI5) dataset..
# for use in the ResidualBind-32 (GOPHER) workflow. The window around each..
# 600-bp CAGI5 locus must be extended on both sides to fill in the 2048-bp..
# ResidualBind-32 model requirements per input sequence
# =============================================================================
# Instructions: Run via 'python preprocessing.py'
#               Output: CAGI5_loci_positions.csv
# =============================================================================


import os, sys
sys.dont_write_bytecode = True
import numpy as np
import h5py
import pandas as pd

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
greatGrandParentDir = os.path.dirname(grandParentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
sys.path.append(greatGrandParentDir)

import squid.utils as squid_utils

loci = ['LDLR','ZFAND3','F9','SORT1','TERT-HEK293T', 
        'MSMB', 'IRF6', 'TERT-GBM', 'IRF4','PKLR',
        'HBB','HBG1','HNF4A','MYCrs6983267','GP1BB']
# see https://kircherlab.bihealth.org/satMutMPRA/
cell_line = ['HepG2', 'MIN6', 'HepG2', 'HepG2', 'HEK293T,SF7996',
       'HEK293T', 'HaCaT', 'HEK293T,SF7996', 'SK-MEL-28', 'K562',
       'HEL 92.1.7', 'HEL 92.1.7', 'HEK293T', 'HEK293T', 'HEL 92.1.7']
# corresponding class indices for GOPHER model
cell_line_idx = [2, 'NaN', 2, 2, 'NaN',
           'NaN', 'NaN', 'NaN', 'NaN', 5,
           'NaN', 'NaN', 'NaN', 'NaN', 'NaN']

df_loci = pd.DataFrame(columns = ['seq_idx', 'locus_name', 'cell_line', 'class_idx', 'locus_center', 'locus_start', 'locus_stop', 'locus_len', 'dataset', 'locus_wt'])
                                  
for idx, locus in enumerate(loci):
    cagiScores = pd.read_csv('CAGI5_info/saturation_scores/combined_%s.tsv' % locus, header=6, delimiter='\t')
    locus_first = cagiScores.iloc[0]['Pos']
    locus_end = cagiScores.iloc[-1]['Pos']
    L = cagiScores.iloc[-1]['Pos'] - cagiScores.iloc[0]['Pos']
    print('Idx: %s, Locus: %s, L: %s' % (idx, locus, L))

    # get one hot encoding (2048-bp region surrounding each 600-bp CAGI5 site)
    with h5py.File(os.path.join('CAGI5_onehots_centered_2048.h5'), 'r') as dataset:
        OHs_wt = np.array(dataset['reference']).astype(np.float32) #shape : (15, 2048, 4)
    OH_wt = OHs_wt[idx] #see 'centered_combined_cagi.bed for information on indexing

    centers = pd.read_csv('CAGI5_info/centered_combined_cagi.csv', header=0, delimiter=',', index_col=0)
    center = centers.loc[centers['locus'] == locus]['center'][0] #center_0 is the center of the CAGI locus

    locus_diff = center - locus_first + 1
    maxL = 2048 #maximum length of GOPHER sequence
    locus_start = int(maxL/2) - locus_diff

    if 0:
        print(squid_utils.oh2seq(OH_wt[locus_start:locus_start+10], alphabet=['A','C','G','T']))
        #print(OH_wt.shape)
        #print(squid_utils.oh2seq(OH_wt, alphabet=['A','C','G','T']))

    df_loci.at[idx, 'seq_idx'] = idx
    df_loci.at[idx, 'locus_name'] = locus
    df_loci.at[idx, 'cell_line'] = cell_line[idx]
    df_loci.at[idx, 'class_idx'] = cell_line_idx[idx]
    df_loci.at[idx, 'locus_center'] = locus_start+int(np.ceil(L/2))
    df_loci.at[idx, 'locus_start'] = locus_start
    df_loci.at[idx, 'locus_stop'] = locus_start+L
    df_loci.at[idx, 'locus_len'] = L
    df_loci.at[idx, 'dataset'] = 'combined'
    df_loci.at[idx, 'locus_wt'] = squid_utils.oh2seq(OH_wt[locus_start:locus_start+L], alphabet=['A','C','G','T'])


# save dataframe to file:
if 1:
    df_loci.to_csv(os.path.join(parentDir,'b_recognition_sites/CAGI5-GOPHER_positions.csv'), index=False)
    print('CAGI5-GOPHER_positions saved to file.')

