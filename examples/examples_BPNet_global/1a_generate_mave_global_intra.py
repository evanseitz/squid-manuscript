# =============================================================================
# Script for an example use of global surrogate modeling for a single motif..
# ..of interest (i.e., disregarding inter-motif relationships)
# (much less optimized than the main SQUID scripts in the parent directory)
# =============================================================================
# To use, first source the proper environment (i.e., 'conda activate bpnet')..
# ..select user settings below, and run via: python 1a_generate_mave_global_intra.py
# =============================================================================

import os, sys
import kipoi
import pybedtools
from pybedtools import BedTool
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import matplotlib.pyplot as plt
import pandas as pd
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandparentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandparentDir)
import squid.utils as squid_utils

# user settings:
N_BGs = 100 #number of mutagenized backgrounds (sequences) to average over to retrieve denoised prediction for a given motif
N = 100000 #total number of mutagenized sequences in MAVE (MPRA) dataset
mut_prob = 0.10 #probability of mutation for each of the N sequences
if 1:
    site = 'TTTGCAT' #'TTTGCATAA' 
    motif_name = 'Oct4'
elif 0:
    site = 'GAACAATAG' #'TTGTTC' 
    motif_name = 'Sox2'
elif 0:
    site = 'GGGTGTGGC'
    motif_name = 'Klf4'
elif 0:
    site = 'AGCCATCAA' #'GCAATCA'
    motif_name = 'Nanog'


    

# model info 
seq_length = 1000 #BPNet input length
alphabet = ['A','C','G','T']
site_length = len(site)
model = kipoi.get_model('BPNet-OSKN')
chromo = 'chr1' #arbitrary
dataDir = os.path.join(pyDir, 'outputs/%s' % motif_name)
if not os.path.exists(dataDir):
    os.makedirs(dataDir)

# compute average profile for multiple random trials of a single motif instance:
predWT_TF_p = np.zeros(shape=(seq_length))
predWT_TF_n = np.zeros(shape=(seq_length))

for z in range(N_BGs):
    randomDNA = random.choices("ACGT", k=seq_length)

    motif_idx = 0
    rand_idx = 0
    synthDNA = []
    for i in range(seq_length):
        if 500 <= i < (500 + site_length):
            synthDNA.append(site[motif_idx])
            motif_idx += 1
        else:
            synthDNA.append(randomDNA[rand_idx])
            rand_idx += 1        
    synthDNA = ''.join(synthDNA)

    oh = squid_utils.seq2oh(synthDNA, alphabet)
    pred = model.predict(np.expand_dims(oh,0))

    predWT_TF_p += pred[motif_name+'/profile'][0][:,0]
    predWT_TF_n += pred[motif_name+'/profile'][0][:,1]

predWT_TF_p /= N_BGs
predWT_TF_n /= N_BGs


# find position where averaged profile is maximum
posMax = np.amax(predWT_TF_p)
negMax = np.amax(predWT_TF_n)
posMaxIdx = np.where(predWT_TF_p == posMax)[0][0]
negMaxIdx = np.where(predWT_TF_n == negMax)[0][0]

# summit max prediction:
summitMax_wt_pos = predWT_TF_p[posMaxIdx]
summitMax_wt_neg = predWT_TF_n[negMaxIdx]

if 1:
    fig, ax = plt.subplots(figsize=(10,3))
    plt.title(site)
    plt.plot(predWT_TF_p)
    plt.plot(-1.*predWT_TF_n)
    plt.xlim(450,550)
    ylim1, ylim2 = ax.get_ylim()
    plt.ylim(ylim1+(.5*(-1.*ylim1+ylim1))), ylim2+(.5*(-1.*ylim1+ylim2))
    plt.ylabel('Average %s pred' % motif_name)
    plt.axhline(0, c='k', linewidth=.5)
    plt.axvspan(495, 504, alpha=0.05, color='k', zorder=-10, label='binding site')
    plt.legend(loc='upper right')
    #ax.axvline(posMaxIdx, color='red', linestyle='-')
    plt.tight_layout()
    #plt.show()

    fig.savefig(os.path.join(dataDir, '%s_avgProfile.png' % motif_name), facecolor="w", dpi=600)


def mutate(WT, mut_rate):
    # mutate sequences for in-silico MPRA:
    mut_seq = []
    for nt in WT: #for each nucleotide in the WT sequence
        if nt == 'A':
            options = ['T', 'G', 'C']
        elif nt == 'T':
            options = ['A', 'G', 'C']
        elif nt == 'G':
            options = ['A', 'T', 'C']
        elif nt == 'C':
            options = ['A', 'T', 'G']
        rand1 = np.random.uniform(0,1)
        rand2 = np.random.uniform(0,1)
        if rand1 < mut_rate:
            if rand2 < 1/3.:
                mut_seq.append(options[0])
            elif 1/3. <= rand2 < 2/3.:
                mut_seq.append(options[1])
            else:
                mut_seq.append(options[2])
        else: #no mutation
            mut_seq.append(nt)            
    return mut_seq


# generate MAVE dataset (MPRA):
mpra_dataset = pd.DataFrame(columns = ['y', 'x'], index=range(N))

print('MAVE progress:')
for n in range(N):
    if n % 1000 == 0:
        print('N =',n)
    randomDNA = random.choices("ACGT", k=seq_length)

    mut_motif = mutate(site, mut_prob)
    mut_motif = ''.join(mut_motif)
    
    motif_idx = 0
    rand_idx = 0
    synthDNA = []
    for i in range(seq_length):
        if 500 <= i < (500 + site_length):
            synthDNA.append(mut_motif[motif_idx])
            motif_idx += 1
        else:
            synthDNA.append(randomDNA[rand_idx])
            rand_idx += 1        
    synthDNA = ''.join(synthDNA)
    
    # Input 1-kb synthetic data into model for prediction:
    '''ofile = open("fasta_file_motif.txt", "w")
    ofile.write(">" + chromo + "\n" + synthDNA)
    ofile.close()
    fasta_file_motif = os.getcwd() + '/fasta_file_motif.txt'

    dl_kwargs = {'intervals_file': intervals_file_motif, 'fasta_file': fasta_file_motif}
    dl = model.default_dataloader(**dl_kwargs)
    batch_iterator = dl.batch_iter(batch_size=4)

    pred_mut_MPRA = model.pipeline.predict(dl_kwargs, batch_size=4)'''

    oh = squid_utils.seq2oh(synthDNA, alphabet)
    pred_mut_MPRA = model.predict(np.expand_dims(oh,0))
    
    predMut_TF_p = pred_mut_MPRA[motif_name+'/profile'][0][:,0] #strand '+'
    predMut_TF_n = pred_mut_MPRA[motif_name+'/profile'][0][:,1] #strand '-'
    
    posMax = np.amax(predMut_TF_p)
    negMax = np.amax(predMut_TF_n)
    posMaxIdx = np.where(predMut_TF_p == posMax)[0][0]
    negMaxIdx = np.where(predMut_TF_n == negMax)[0][0]

    # summit max prediction:
    summitMax_mut_pos = predMut_TF_p[posMaxIdx]
    summitMax_mut_neg = predMut_TF_n[negMaxIdx]
    summitMax_foldChange = summitMax_mut_pos / summitMax_wt_pos
    
    mpra_dataset = mpra_dataset.append({'y' : summitMax_foldChange, 
                                        'x' : synthDNA},
                                       ignore_index = True)
    
# ready MPRA for training
final_df = mpra_dataset.copy()
final_df.drop_duplicates(subset='x', keep=False, inplace=True)
final_df['set'] = np.random.choice(a=['training','test','validation'],
                                   p=[.6,.2,.2],
                                   size=len(final_df))
# rearrange columns
new_cols = ['set'] + list(final_df.columns[0:-2]) + ['x']
final_df = final_df[new_cols]

if 1: #save to file
    final_df.to_csv(os.path.join(dataDir, 'MPRA_%s_N%s.csv.gz' % (motif_name, N)), index=False, compression='gzip')

final_df.head()

'''if os.path.exists(os.path.join(pyDir, 'fasta_file_motif.txt')):
    os.remove(os.path.join(pyDir,'fasta_file_motif.txt'))
if os.path.exists(os.path.join(pyDir, 'fasta_file_motif.txt.fai')):
    os.remove(os.path.join(pyDir,'fasta_file_motif.txt.fai'))
if os.path.exists(os.path.join(pyDir, 'intervals_file_motif.bed')):
    os.remove(os.path.join(pyDir,'intervals_file_motif.bed'))'''