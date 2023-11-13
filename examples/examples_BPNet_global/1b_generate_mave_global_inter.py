# =============================================================================
# Script for an example use of global surrogate modeling for a pair of motifs..
# ..such that inter-motif relationships can be analyzed
# (much less optimized than the main SQUID scripts in the parent directory)
# =============================================================================
# To use, first source the proper environment (i.e., 'conda activate bpnet')..
# ..select user settings below, and run via (e.g.): 
#       python 1b_generate_mave_global_inter.py 5
# ..where the positive integer 5 represents the desired inter-motif distance.
# Alternatively, a series of inter-motif instances can be run as a batch via:
#       bash 1b_batch.sh
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings


def op(pyDir, inter_dist): #'inter_dist' : distance between motifs (see below)

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
    N = 200000 #total number of mutagenized sequences in MAVE (MPRA) dataset
    mut_prob = 0.15 #probability of mutation for each of the N sequences
    siteA = 'TAAAAAGAGCAATCAA' #len=16; final 'T' is center
    motifA_name = 'Nanog'
    siteB = 'AGAACAATAGAG' #len=12; 7th position ('A') is center
    motifB_name = 'Sox2'


    # model info 
    seq_length = 1000 #BPNet input length
    alphabet = ['A','C','G','T']
    siteA_length = len(siteA)
    siteB_length = len(siteB)
    model = kipoi.get_model('BPNet-OSKN')
    chromo = 'chr1' #arbitrary
    dataDir = os.path.join(pyDir, 'outputs/%s%s' % (motifA_name, motifB_name))
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    profDir = os.path.join(dataDir, 'avg_profiles')
    if not os.path.exists(profDir):
        os.mkdir(profDir)
    mpraDir = os.path.join(dataDir, 'MPRA')
    if not os.path.exists(mpraDir):
        os.mkdir(mpraDir)

    # compute average profile for multiple random trials of a single motif_A instance:
    predWT_TF_p = np.zeros(shape=(seq_length))
    predWT_TF_n = np.zeros(shape=(seq_length))
    # repeat for motif_B:
    predWT_ref_p = np.zeros(shape=(seq_length))
    predWT_ref_n = np.zeros(shape=(seq_length))


    for z in range(N_BGs):
        randomDNA = random.choices("ACGT", k=seq_length)

        motifA_idx = 0
        motifB_idx = 0
        rand_idx = 0
        synthDNA = []
        for i in range(seq_length):
            if (490-8) <= i < (490+8): #center at 495 for Nanog
                synthDNA.append(siteA[motifA_idx])
                motifA_idx += 1
            elif ((495+inter_dist+1)-6) <= i < ((495+inter_dist+1)+6): #Sox2 variable distance away
                synthDNA.append(siteB[motifB_idx])
                motifB_idx += 1
            else:
                synthDNA.append(randomDNA[rand_idx])
                rand_idx += 1        
        synthDNA = ''.join(synthDNA)

        # input 1-kb synthetic data into model for prediction:
        '''ofile = open('fasta_file_motif_dist%s.txt' % inter_dist, 'w')
        ofile.write(">" + chromo + "\n" + synthDNA)
        ofile.close()
        fasta_file_motif = os.getcwd() + '/fasta_file_motif_dist%s.txt' % inter_dist

        intervals_file_bed = pybedtools.BedTool(f"{chromo} 0 %s" % seq_length, from_string=True)
        intervals_file_bed.saveas('intervals_file_motif_dist%s.bed' % inter_dist)
        intervals_file_motif = os.getcwd() + '/intervals_file_motif_dist%s.bed' % inter_dist

        dl_kwargs = {'intervals_file': intervals_file_motif, 'fasta_file': fasta_file_motif}
        dl = model.default_dataloader(**dl_kwargs)
        batch_iterator = dl.batch_iter(batch_size=4)

        pred = model.pipeline.predict(dl_kwargs, batch_size=4)'''

        oh = squid_utils.seq2oh(synthDNA, alphabet)
        pred = model.predict(np.expand_dims(oh,0))

        predWT_TF_p += pred[motifA_name+'/profile'][0][:,0]
        predWT_TF_n += pred[motifA_name+'/profile'][0][:,1]
        predWT_ref_p += pred[motifB_name+'/profile'][0][:,0]
        predWT_ref_n += pred[motifB_name+'/profile'][0][:,1]

    predWT_TF_p /= N_BGs
    predWT_TF_n /= N_BGs
    predWT_ref_p /= N_BGs
    predWT_ref_n /= N_BGs


    # find position where averaged profile is maximum
    posMaxA = np.amax(predWT_TF_p[490-35:490+35])
    negMaxA = np.amax(predWT_TF_n[490-35:490+35])
    posMaxIdx = np.where(predWT_TF_p == posMaxA)[0][0]
    negMaxIdx = np.where(predWT_TF_n == negMaxA)[0][0]

    # summit max prediction:
    summitMax_wt_pos = predWT_TF_p[posMaxIdx]
    summitMax_wt_neg = predWT_TF_n[negMaxIdx]
    # summit avg prediction:
    summitAvg_wt_p = np.mean(predWT_TF_p[posMaxIdx-10:posMaxIdx+10])

    if 1: #plot prediction profile for each motif
        fig, ax = plt.subplots(2)
        ax[0].plot(predWT_TF_p)
        ax[0].plot(-1.*predWT_TF_n)
        ax[0].set_xlim(450,495+inter_dist+35)
        #ax[0].axvline(495, color='gray', linestyle='-', linewidth=1)
        ax[0].axvline(posMaxIdx, color='lightgray', linestyle='--', linewidth=1, zorder=-10)
        #ax[0].axvline(negMaxIdx, color='lightgray', linestyle='--', linewidth=1, zorder=-10)
        ax[0].axvspan(482, 498, alpha=0.10, color='green', zorder=-10, label=motifA_name)
        ax[0].axvspan((495+inter_dist+1)-6, (495+inter_dist+1)+6, alpha=0.10, color='red', zorder=-10, label=motifB_name)
        ax[0].set_title('%s –– %s | dist = %s' % (motifA_name, motifB_name, inter_dist))
        ax[0].set_ylabel('%s Pred' % motifA_name)
        ax[0].axhline(0, c='k', linewidth=.5, zorder=-11)
        ax[0].set_xlim(450,540)
       #ax[0].set_ylim(-8,8) #ad hoc
        ax[0].legend(loc='lower left', frameon=False)

        ax[1].plot(predWT_ref_p)
        ax[1].plot(-1.*predWT_ref_n)
        ax[1].set_xlim(450,495+inter_dist+35)
        ax[1].axvspan(482, 498, alpha=0.10, color='green', zorder=-10, label=motifA_name)
        ax[1].axvspan((495+inter_dist+1)-6, (495+inter_dist+1)+6, alpha=0.10, color='red', zorder=-10, label=motifB_name)
        ax[1].set_ylabel('%s Pred' % motifB_name)
        ax[1].axhline(0, c='k', linewidth=.5, zorder=-11)
        ax[1].set_xlim(450,540)
        #ax[1].set_ylim(-.8,.8) #ad hoc
        ax[1].legend(loc='lower left', frameon=False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(profDir, 'avgProfile_%s%s_dist%s.png' % (motifA_name, motifB_name, inter_dist)), facecolor="w", dpi=600)
        #plt.show()
        plt.close()


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

        mut_motifA = mutate(siteA, mut_prob)
        mut_motifA = ''.join(mut_motifA)
        mut_motifB = mutate(siteB, mut_prob)
        mut_motifB = ''.join(mut_motifB)

        motifA_idx = 0
        motifB_idx = 0
        rand_idx = 0
        synthDNA = []
        for i in range(seq_length):
            if (490-8) <= i < (490+8): #center at 495 for Nanog1
                synthDNA.append(mut_motifA[motifA_idx])
                motifA_idx += 1
            elif ((495+inter_dist+1)-6) <= i < ((495+inter_dist+1)+6): #Sox2 variable distance away
                synthDNA.append(mut_motifB[motifB_idx])
                motifB_idx += 1
            else:
                synthDNA.append(randomDNA[rand_idx])
                rand_idx += 1
        synthDNA = ''.join(synthDNA)
        
        # Input 1-kb synthetic data into model for prediction:
        '''ofile = open("fasta_file_motif_%s.txt" % inter_dist, "w")
        ofile.write(">" + chromo + "\n" + synthDNA)
        ofile.close()
        fasta_file_motif = os.getcwd() + '/fasta_file_motif_%s.txt' % inter_dist

        dl_kwargs = {'intervals_file': intervals_file_motif, 'fasta_file': fasta_file_motif}
        dl = model.default_dataloader(**dl_kwargs)
        batch_iterator = dl.batch_iter(batch_size=4)

        pred_mut_MPRA = model.pipeline.predict(dl_kwargs, batch_size=4)'''

        oh = squid_utils.seq2oh(synthDNA, alphabet)
        pred_mut_MPRA = model.predict(np.expand_dims(oh,0))
        
        predMut_TF_p = pred_mut_MPRA[motifA_name+'/profile'][0][:,0] #strand '+'
        predMut_TF_n = pred_mut_MPRA[motifA_name+'/profile'][0][:,1] #strand '-'
        
        posMax = np.amax(predMut_TF_p)
        negMax = np.amax(predMut_TF_n)
        posMaxIdx = np.where(predMut_TF_p == posMax)[0][0]
        negMaxIdx = np.where(predMut_TF_n == negMax)[0][0]

        if 0: #summit max prediction
            summitMax_mut_pos = predMut_TF_p[posMaxIdx]
            summitMax_mut_neg = predMut_TF_n[negMaxIdx]
            foldChange = summitMax_mut_pos / summitMax_wt_pos
        else: #summit avg prediction
            summitAvg_mut_p = np.mean(predMut_TF_p[posMaxIdx-10:posMaxIdx+10])
            summitAvg_mut_n = np.mean(predMut_TF_n[negMaxIdx-10:negMaxIdx+10])
            foldChange = summitAvg_mut_p / summitAvg_wt_p
        
        mpra_dataset = mpra_dataset.append({'y' : foldChange,
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

    print(final_df.head())

    if 1: #save to file
        final_df.to_csv(os.path.join(mpraDir, 'MPRA_%s%s_N%s_dist%s.csv.gz' % (motifA_name, motifB_name, N, inter_dist)), index=False, compression='gzip')

    '''if os.path.exists(os.path.join(pyDir, 'fasta_file_motif_%s.txt' % inter_dist)):
        os.remove(os.path.join(pyDir,'fasta_file_motif_%s.txt' % inter_dist))
    if os.path.exists(os.path.join(pyDir, 'fasta_file_motif_%s.txt.fai' % inter_dist)):
        os.remove(os.path.join(pyDir,'fasta_file_motif_%s.txt.fai' % inter_dist))
    if os.path.exists(os.path.join(pyDir, 'intervals_file_motif_%s.bed' % inter_dist)):
        os.remove(os.path.join(pyDir,'intervals_file_motif_%s.bed' % inter_dist))'''
    
if __name__ == '__main__':
    path1 = os.path.dirname(os.path.abspath(__file__))

    """
    df_idx : INT >= 0
        inter-motif distance between two motifs
    """
    
    if len(sys.argv) > 1:
        inter_dist = int(sys.argv[1])

    else:
        print('')
        print('Script must be run with trailing index argument:')
        print('e.g., 1b_generate_mave_global_inter.py 5')
        print('')
        sys.exit(0)
    op(path1, inter_dist)