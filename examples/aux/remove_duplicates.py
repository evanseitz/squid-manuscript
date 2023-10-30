# =============================================================================
# Numerous (>15%) inputs in the BPNet test set are identical in both sequence..
# ..composition X and activity Y. This script removes all of these duplicates..
# ..and should be used for other problematic datasets as needed
# =============================================================================
# Instructions: Before running, make sure to source the correct environment in..
#               ..the CLI. Next, customize the variables in the user inputs below..
#               ..to match the desired file(s) produced by '1_locate_patterns.py'
#               The current script can be run via: python remove_duplicates.py
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
import pandas as pd

pyDir = os.path.dirname(os.path.abspath(__file__))
parDir = os.path.dirname(pyDir)


# =============================================================================
# User inputs
# =============================================================================
motif_A_name = 'Sox2'
motif_B_name = None#'Oct4' #if None, removing duplicates for motif/motif pairs will be skipped
example = 'BPNet'#'GOPHER'#'BPNet'
model_name = 'model_BPNet_OSKN' #'model_ResidualBind32_ReLU'#'model_BPNet_OSKN'

# =============================================================================
# Load data
# =============================================================================
userDir = os.path.join(parDir, 'examples_%s/b_recognition_sites/%s' % (example, model_name))
motif_singles = pd.read_csv(os.path.join(userDir, '%s_positions.csv' % (motif_A_name)))
if motif_B_name is not None:
    motif_pairs = pd.read_csv(os.path.join(userDir,'%s_%s_positions.csv' % (motif_A_name, motif_B_name)))


# =============================================================================
# Recreate dataframe(s) without duplicate entries
# =============================================================================
if 1:
    df_motif_A = pd.DataFrame(columns = ['seq_idx', 'motif_rank', 'motif_wt', 'motif_mutations', 'motif_start'])
    
    # dataframe needs to be ordered by number of mutations
    motif_singles = motif_singles.sort_values(by = ['motif_mutations', 'motif_rank'], ascending = [True, False])
    motif_singles.reset_index(drop=True,inplace=True)
    if motif_B_name is not None:
        motif_pairs = motif_pairs.sort_values(by = ['motif_mutations', 'motif_rank'], ascending = [True, False])
        motif_pairs.reset_index(drop=True,inplace=True)
    
    # recreate motif A dataframe without duplicate entries:
    if motif_A_name is not None:
        df_motif_A.at[0, 'seq_idx'] = motif_singles['seq_idx'][0]
        df_motif_A.at[0, 'motif_rank'] = motif_singles['motif_rank'][0]
        df_motif_A.at[0, 'motif_wt'] = motif_singles['motif_wt'][0]
        df_motif_A.at[0, 'motif_mutations'] = motif_singles['motif_mutations'][0]
        df_motif_A.at[0, 'motif_start'] = motif_singles['motif_start'][0]
        
        print('%s progress:' % (motif_A_name))
        for i in range(1,len(motif_singles)):
            if motif_singles['motif_start'][i] != motif_singles['motif_start'][i-1]:
                df_motif_A.at[i, 'seq_idx'] = motif_singles['seq_idx'][i]
                df_motif_A.at[i, 'motif_rank'] = motif_singles['motif_rank'][i]
                df_motif_A.at[i, 'motif_wt'] = motif_singles['motif_wt'][i]
                df_motif_A.at[i, 'motif_mutations'] = motif_singles['motif_mutations'][i]
                df_motif_A.at[i, 'motif_start'] = motif_singles['motif_start'][i]
            if i % 10000 == 0:
                print(i)
                
        df_motif_A = df_motif_A.sort_values(by = ['motif_rank'], ascending = [False])
        
        os.rename(os.path.join(userDir,'%s_positions.csv' % (motif_A_name)), os.path.join(userDir,'%s_positions_dups.csv' % (motif_A_name)))
        df_motif_A.to_csv(os.path.join(userDir, '%s_positions.csv' % motif_A_name), index=False)
    
        print('%s_positions with duplicates renamed to %s_positions_dups.' % (motif_A_name, motif_A_name))
        print('%s_positions without duplicates saved to file as %s_positions.' % (motif_A_name, motif_A_name))

if 0:
    # recreate motif A/B dataframe without duplicate entries:
    if motif_B_name is not None:
        df_motifs_AB = pd.DataFrame(columns = ['seq_idx', 'motif_rank', 'motif_wt', 'motif_mutations', 'motif_start', 'inter_dist'])

        df_motifs_AB.at[0, 'seq_idx'] = motif_pairs['seq_idx'][0]
        df_motifs_AB.at[0, 'motif_rank'] = motif_pairs['motif_rank'][0]
        df_motifs_AB.at[0, 'motif_wt'] = motif_pairs['motif_wt'][0]
        df_motifs_AB.at[0, 'motif_mutations'] = motif_pairs['motif_mutations'][0]
        df_motifs_AB.at[0, 'motif_start'] = motif_pairs['motif_start'][0]
        df_motifs_AB.at[0, 'inter_dist'] = motif_pairs['inter_dist'][0]
                
        print('%s_%s progress:' % (motif_A_name, motif_B_name))
        for i in range(1,len(motif_pairs)):
            if motif_pairs['motif_start'][i] != motif_pairs['motif_start'][i-1]:
                df_motifs_AB.at[i, 'seq_idx'] = motif_pairs['seq_idx'][i]
                df_motifs_AB.at[i, 'motif_rank'] = motif_pairs['motif_rank'][i]
                df_motifs_AB.at[i, 'motif_wt'] = motif_pairs['motif_wt'][i]
                df_motifs_AB.at[i, 'motif_mutations'] = motif_pairs['motif_mutations'][i]
                df_motifs_AB.at[i, 'motif_start'] = motif_pairs['motif_start'][i]
                df_motifs_AB.at[i, 'inter_dist'] = motif_pairs['inter_dist'][i]
            if i % 10000 == 0:
                print(i)
                
        df_motifs_AB = df_motifs_AB.sort_values(by = ['motif_rank'], ascending = [False])
        
        os.rename(os.path.join(userDir,'%s_%s_positions.csv' % (motif_A_name, motif_B_name)), os.path.join(userDir,'%s_%s_positions_dups.csv' % (motif_A_name, motif_B_name)))
        df_motifs_AB.to_csv(os.path.join(userDir, '%s_%s_positions.csv' % (motif_A_name, motif_B_name)), index=False)
        
        print('%s_%s_positions with duplicates renamed to %s_%s_positions_dups.' % (motif_A_name, motif_B_name, motif_A_name, motif_B_name))
        print('%s_%s_positions without duplicates saved to file as %s_%s_positions.' % (motif_A_name, motif_B_name, motif_A_name, motif_B_name))