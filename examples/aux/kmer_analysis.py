# =============================================================================
# Conduct kmer analysis of all recogntion sites found via '1_locate_patterns.py'
# ..up to the given number of mutations explored (defined in 'set_parameters.py')
# For each permutation, the frequency of occurance in the test set as well as..
# ..its average activity will be calculated.
# =============================================================================
# Instructions: Before running, make sure to source the correct environment in..
#               ..the CLI. Next, customize the variables in the user inputs below..
#               ..to match the desired file(s) produced by '1_locate_patterns.py'
#               The current script can be run via: python kmer_analysis.py
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

pyDir = os.path.dirname(os.path.abspath(__file__))
parDir = os.path.dirname(pyDir)


# =============================================================================
# User inputs
# =============================================================================
motif_A = 'ATGCANA'
motif_A_name = 'Oct4'
example = 'BPNet'
model_name = 'model_BPNet_OSKN'
a = 'ACGT' #alphabet

# =============================================================================
# Load data
# =============================================================================
userDir = os.path.join(parDir, 'examples_%s/b_recognition_sites/%s' % (example, model_name))
motif_singles = pd.read_csv(os.path.join(userDir, '%s_positions.csv' % (motif_A_name)))


# =============================================================================
# k-mer analysis
# =============================================================================
L = len(motif_A)
kmer_df = pd.DataFrame(columns = ['k', 'y', 'f'], index=range(4**L))
idx = 0
for kmer in itertools.product(a, repeat=L):
    kmer_df.at[idx,'k'] = ''.join(kmer)
    idx += 1
    
for i in range(0,len(motif_singles)):
    wt = motif_singles['motif_wt'][i]
    kmer_idx = kmer_df.index[kmer_df['k'] == wt]
    if np.isnan(kmer_df.at[kmer_idx[0],'f']):
        kmer_df.at[kmer_idx[0],'f'] = 0
        kmer_df.at[kmer_idx[0],'y'] = 0
    kmer_df.at[kmer_idx[0],'f'] += 1
    kmer_df.at[kmer_idx[0],'y'] += motif_singles['motif_rank'][i]

kmer_df = kmer_df.dropna(subset=['f'])
kmer_df['yavg'] = kmer_df['y']/kmer_df['f']

kmer_df['f'].plot(kind='bar', use_index='True')
plt.title('Oct4')
plt.xlabel('kmer index')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig(os.path.join(userDir, '%s_kmer_freq.png' % (motif_A_name)), facecolor='w', dpi=200)
plt.close()

kmer_df['yavg'].plot(kind='bar', use_index='True')
plt.title('Oct4')
plt.xlabel('kmer index')
plt.ylabel('avg(y)')
plt.ylim(kmer_df['yavg'].min())
plt.tight_layout()
plt.savefig(os.path.join(userDir, '%s_kmer_y.png' % (motif_A_name)), facecolor='w', dpi=200)

kmer_df = kmer_df.sort_values(by = ['yavg'], ascending = [False])
kmer_df.to_csv(os.path.join(userDir, '%s_kmers_sort_y.csv' % motif_A_name))#, index=False)

kmer_df = kmer_df.sort_values(by = ['f'], ascending = [False])
kmer_df.to_csv(os.path.join(userDir, '%s_kmers_sort_f.csv' % motif_A_name))#, index=False)
print('%s_kmers saved to file.' % motif_A_name)