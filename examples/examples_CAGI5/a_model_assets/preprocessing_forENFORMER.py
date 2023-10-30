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
from Bio import SeqIO
import tensorflow as tf


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
# corresponding class strings for ENFORMER model
cell_line_idx = ['HepG2', 'pancreas', 'HepG2', 'HepG2', 'HEK293',
           'HEK293', 'keratinocyte', 'glioblastoma', 'SK-MEL', 'K562',
           'K562', 'K562', 'HEK293', 'HEK293', 'K562']

df_loci = pd.DataFrame(columns = ['seq_idx', 'locus_name', 'cell_line', 'class_idx', 'locus_center', 'locus_start', 'locus_stop', 'locus_len', 'dataset', 'locus_wt'])
maxL = 393216 #maximum length of ENFORMER sequence
X = np.zeros(shape=(len(loci), maxL, 4))

for idx, locus in enumerate(loci):
    cagiScores = pd.read_csv('CAGI5_info/saturation_scores/combined_%s.tsv' % locus, header=6, delimiter='\t')
    locus_first = cagiScores.iloc[0]['Pos']
    locus_last = cagiScores.iloc[-1]['Pos']
    L = cagiScores.iloc[-1]['Pos'] - cagiScores.iloc[0]['Pos']
    print('Idx: %s, Locus: %s, L: %s' % (idx, locus, L))
    chr = 'chr%s' % cagiScores.iloc[0]['#Chrom']

    centers = pd.read_csv('CAGI5_info/centered_combined_cagi.csv', header=0, delimiter=',', index_col=0)
    center = centers.loc[centers['locus'] == locus]['center'][0] #center_0 is the center of the CAGI locus

    locus_diff = center - locus_first + 1
    locus_start = int(maxL/2) - locus_diff

    fastaFile = os.path.join(grandParentDir,'examples_ENFORMER/a_model_assets/hg19/%s.fa' % chr)
    for seq_record in SeqIO.parse(fastaFile, "fasta"):
        print(str(seq_record.id))
        locus_seq = str(seq_record.seq[locus_first-1:locus_first-1+L]).upper()
        full_seq = str(seq_record.seq[locus_first-locus_start-1:locus_first-locus_start-1+maxL]).upper()
        #print(locus_seq)

    if 1: #append one-hot encodings to matrix
        X[idx,:,:] = squid_utils.seq2oh(full_seq, alphabet=['A','C','G','T'])

    df_loci.at[idx, 'seq_idx'] = idx
    df_loci.at[idx, 'locus_name'] = locus
    df_loci.at[idx, 'cell_line'] = cell_line[idx]
    df_loci.at[idx, 'class_idx'] = cell_line_idx[idx]
    df_loci.at[idx, 'locus_center'] = locus_start+int(np.ceil(L/2))
    df_loci.at[idx, 'locus_start'] = locus_start
    df_loci.at[idx, 'locus_stop'] = locus_start+L
    df_loci.at[idx, 'locus_len'] = L
    df_loci.at[idx, 'dataset'] = 'combined'
    df_loci.at[idx, 'locus_wt'] = locus_seq


if 1: #save dataframe to file
    df_loci.to_csv(os.path.join(parentDir,'b_recognition_sites/CAGI5-ENFORMER_positions.csv'), index=False)
    print('CAGI5-ENFORMER_positions saved to file.')

if 1: #save one-hot encodings to file
    with h5py.File(os.path.join(parentDir,'a_model_assets/CAGI5_onehots_centered_%s.h5' % maxL), 'w') as hf:
        hf.create_dataset('reference', data=X)
    
    if 0: #sanity check
        with h5py.File(os.path.join(parentDir,'a_model_assets/CAGI5_onehots_centered_%s.h5' % maxL), 'r') as hf:
            data = hf['reference'][:]
        print(locus_seq[:10])
        print(data[0,locus_start:locus_start+10,:])
