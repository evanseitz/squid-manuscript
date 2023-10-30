import os, sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker
from scipy import signal, stats
from operator import itemgetter
import h5py
from Bio import motifs #pip install biopython

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
import squid.utils as squid_utils

# used for generating data needed by and compiled within Fig. 4bc
# turn on each of the switches below (independently per run) to generate data for a specific model/motif combination
# environment: e.g., 'mavenn'

# =============================================================================
# User inputs
# =============================================================================
pwm_file = False
save = True #save outputs
alphabet = ['A','C','G','T']
#gauge = 'empirical' #{empirical, hierarchical, wildtype, default}
#gauge = 'hierarchical'
gauge = 'wildtype'
#gauge = 'default'

saveDir = os.path.join(pyDir,'Fig4_data/%s' % gauge)
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# choose only one model/motif below
if 1: #DeepSTARR
    if 1: # DRE
        pfm_fname = os.path.join(parentDir,'examples_DeepSTARR/b_recognition_sites/PWMs/known_PWMs/dre__homer-M00230.pfm') #path to position frequency matrix
        motif_A_name = 'DRE'
        motif_A = 'TATCGATA'
        left_pad, right_pad = 3,1 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        rc = False
    if 0: #Ohler 1
        pfm_fname = os.path.join(parentDir,'examples_DeepSTARR/b_recognition_sites/PWMs/known_PWMs/ohler1__homer-M00232.pfm') #path to position frequency matrix
        motif_A_name = 'Ohler1'
        motif_A = 'AGTGTGACC'
        left_pad, right_pad = 1,2 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        rc = True
    if 0: #AP-1
        pfm_fname = os.path.join(parentDir,'examples_GOPHER/b_recognition_sites/PWMs/known_PWMs/ap1__MA0476.1.pfm') #path to position frequency matrix
        motif_A_name = 'AP1'
        motif_A = 'TGACTCA'
        left_pad, right_pad = 2,2 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        rc = False
        
    analysis_path = os.path.join(parentDir,'examples_DeepSTARR/d_outputs_analysis/model_DeepSTARR') #see below
    other = 'DeepExplainer'
    example = 'DeepSTARR'
    max_muts = 2
    model_name = 'model_DeepSTARR'
    userDir = os.path.join(parentDir, 'examples_%s' % example)
    with h5py.File(os.path.join(userDir, 'a_model_assets/deepstarr_data.h5'), 'r') as dataset:
        X_in = np.array(dataset['x_test']).astype(np.float32)
    
if 0: #ResidualBind32
    if 0: #AP-1
        if 0:
            pfm_fname = os.path.join(parentDir,'examples_GOPHER/b_recognition_sites/PWMs/known_PWMs/ap1__MA0491.1.pfm') #path to position frequency matrix
            rc = True
        if 1:
            pfm_fname = os.path.join(parentDir,'examples_GOPHER/b_recognition_sites/PWMs/known_PWMs/ap1__MA0476.1.pfm') #path to position frequency matrix
            rc = False
        motif_A_name = '13_AP1'
        motif_A = 'TGAGTCA'
        left_pad, right_pad = 2,2 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        class_idx = 13
    if 1: #IRF1
        pfm_fname = os.path.join(parentDir,'examples_GOPHER/b_recognition_sites/PWMs/known_PWMs/irf1__MA0050.1.pfm') #path to position frequency matrix
        rc = False
        motif_A_name = '7_IRF1-long'
        motif_A = 'AANTGAAAC'
        left_pad, right_pad = 2,1 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        class_idx = 7
        
        
    analysis_path = os.path.join(parentDir,'examples_GOPHER/d_outputs_analysis/model_ResidualBind32_ReLU_single') #see below
    other = 'Saliency'
    example = 'GOPHER'
    max_muts = 2
    model_name = 'model_ResidualBind32_ReLU_single'
    userDir = os.path.join(parentDir, 'examples_%s' % example)
    with h5py.File(os.path.join(userDir, 'a_model_assets/cell_line_%s.h5' % class_idx), 'r') as dataset:
        X_in = np.array(dataset['X']).astype(np.float32)
    
if 0: #BPNET 
    if 0: #Oct4-Sox2
        pfm_fname = os.path.join(parentDir,'examples_BPNet/b_recognition_sites/PWMs/known_PWMs/oct4sox2__MA0142.1.pfm') #path to position frequency matrix
        motif_A_name = 'Oct4-Sox2_N'
        motif_A = 'TTNNNATGCAAA'
        left_pad, right_pad = 2,1 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        max_muts = 2
        rc = False
    if 0: #Sox2
        pfm_fname = os.path.join(parentDir, 'examples_BPNet/b_recognition_sites/PWMs/known_PWMs/SOX2.H12CORE.0.P.B_hocomoco.pcm')
        motif_A_name = 'Sox2'
        motif_A = 'GAACAATAG'
        left_pad, right_pad = 2,1 #based on pfm padding around core motif compared to 'motif_A' defined in 'set_parameters.py'
        max_muts = 2
        rc = True
    if 1: #Nanog
        if 0:
            pfm_fname = os.path.join(parentDir,'examples_BPNet/b_recognition_sites/PWMs/known_PWMs/nanog__HUMAN.H11MO.1.B.pcm') #path to position frequency matrix
            left_pad, right_pad = 1,0 #AGCCATCAA
            rc = False
        if 0:
            pfm_fname = os.path.join(parentDir,'examples_BPNet/b_recognition_sites/PWMs/known_PWMs/nanog__mES-Nanog-ChIP-Seq_GSE11724_Homer_Motif224.pfm')
            left_pad, right_pad = 0,1 #AGCCATCAA
            pwm_file = True
            rc = False
        if 1:
            pfm_fname = os.path.join(parentDir, 'examples_BPNet/b_recognition_sites/PWMs/known_PWMs/NANOG.H12CORE.1.P.B.pcm')
            left_pad, right_pad = 1,1 #AGCCATCAA
            rc = True
        motif_A_name = 'Nanog'
        motif_A = 'AGCCATCAA'
        max_muts = 2
        
    analysis_path = os.path.join(parentDir,'examples_BPNet/d_outputs_analysis/model_BPNet_OSKN') #see below
    other = 'DeepLIFT'
    example = 'BPNet'
    model_name = 'model_BPNet_OSKN'
    userDir = os.path.join(parentDir, 'examples_%s' % example)
    with h5py.File(os.path.join(userDir, 'a_model_assets/bpnet_seqs_chr1-8-9.h5'), 'r') as dataset:
        X_in = np.array(dataset['X']).astype(np.float32)
    
model_pad = 50
seq_total = 50
standardize_local = True
scope = 'intra'

motif_A_len = len(motif_A)

# load in directories for singlet attribution maps
if standardize_local is True:
    all_type = 'all_norm'
else:
    all_type = 'all'
all_add_mut0 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/%s_add_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_add_mut1 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut1/pad%s/ADD_A/ADD_%s/%s_add_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_add_mut2 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut2/pad%s/ADD_A/ADD_%s/%s_add_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_ism_mut0 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/%s_ISM_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_ism_mut1 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut1/pad%s/ADD_A/ADD_%s/%s_ISM_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_ism_mut2 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut2/pad%s/ADD_A/ADD_%s/%s_ISM_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_other_mut0 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/%s_other_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_other_mut1 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut1/pad%s/ADD_A/ADD_%s/%s_other_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
all_other_mut2 = np.load(os.path.join(analysis_path, 'SQUID_%s_intra_mut2/pad%s/ADD_A/ADD_%s/%s_other_A.npy' % (motif_A_name, model_pad, gauge, all_type)))
# load in directories for averaged attribution maps (reference to zero core mutations)
avg_add_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_additive_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
avg_ism_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_ISM_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)
avg_other_mut0 = pd.read_csv(os.path.join(analysis_path, 'SQUID_%s_intra_mut0/pad%s/ADD_A/ADD_%s/avg_other_A.csv' % (motif_A_name, model_pad, gauge)), index_col=0)


if 0: #view average of additive attribution maps
    temp1 = squid_utils.arr2pd(all_add_mut0[0,:,:], ['A','C','G','T'])
    fig, ax = plt.subplots()
    logomaker.Logo(df=temp1,
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    ax.set_title('Sequence 1')
    plt.show()
    
    temp2 = squid_utils.arr2pd(all_add_mut0[1,:,:], ['A','C','G','T'])
    fig, ax = plt.subplots()
    logomaker.Logo(df=temp2,
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    ax.set_title('Sequence 2')
    plt.show()
if 0: #view average of additive attribution maps
    fig, ax = plt.subplots()
    logomaker.Logo(df=avg_add_mut0,
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    ax.set_title('Average')
    plt.show()


# =============================================================================
# Compute position specific scoring matrix; i.e., position weight matrix (PWM)
# (see https://en.wikipedia.org/wiki/Position_weight_matrix)
# =============================================================================
if pwm_file is False:
    PFM = motifs.read(open(pfm_fname), 'jaspar') #position frequency matrix
    
    PPM = PFM.counts.normalize(pseudocounts=0.5) #position probability matrix
    pwm = PPM.log_odds() #identical to performing np.log2(np.array(pd.DataFrame(PPM))/.25)
    if rc:
        pwm = pwm.reverse_complement()
    PWM = pd.DataFrame(pwm)
else:
    pwm = pd.read_csv(pfm_fname, header=None, delimiter=r"\s+")
    PWM = squid_utils.arr2pd(np.array(pwm), ['A','C','G','T'])


if 1: #view PWM
    fig, ax = plt.subplots(figsize=(2,1))
    logomaker.Logo(df=PWM,
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    #ax.set_title('PWM')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, 'PWM_drawLogo_%s_%s.pdf' % (example, motif_A_name)), facecolor='w', dpi=200)
    #plt.show()
    plt.close()
    
if save:
    PWM.to_csv(os.path.join(saveDir, 'PWM_logo_%s_%s.csv' % (example, motif_A_name)))
    
    
# =============================================================================
# Compute Euclidean distances of each singlet map to the zero-core-mutation average
# =============================================================================
add_all = np.vstack((all_add_mut0, all_add_mut1, all_add_mut2))
ism_all = np.vstack((all_ism_mut0, all_ism_mut1, all_ism_mut2))
other_all = np.vstack((all_other_mut0, all_other_mut1, all_other_mut2))

add_errors = np.linalg.norm(add_all - np.array(avg_add_mut0), axis=(1,2))
ism_errors = np.linalg.norm(ism_all - np.array(avg_ism_mut0), axis=(1,2))
other_errors = np.linalg.norm(other_all - np.array(avg_other_mut0), axis=(1,2))


# =============================================================================
# Rerun algorithms needed to match setup used in previous script
# =============================================================================
if scope == 'intra':
    #dataDir = os.path.join(userDir, 'c_surrogate_outputs/%s/SQUID_%s_%s_mut%s' % (model_name, motif_A_name, scope, use_mut))
    #saveDir = os.path.join(userDir, 'd_outputs_analysis/%s/SQUID_%s_%s_mut%s_pad%s' % (model_name, motif_A_name, scope, use_mut, fig_pad))
    motif_info = pd.read_csv(os.path.join(userDir,'b_recognition_sites/%s/%s_positions.csv' % (model_name, motif_A_name)))
elif scope == 'inter':
    print('Not the intent of the current analysis...')
    
    
# =============================================================================
# Load and sort motif-location dataframe by mutation count
# =============================================================================
pwm_scores = []
pwm_colors = []
seq_info = []
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for core_muts in range(0,max_muts+1):
    print('Muts: %s' % core_muts)
    motif_info_mut = motif_info.loc[motif_info['motif_mutations'] == core_muts]
    
    motif_info_mut.reset_index(drop=True, inplace=True)
    motif_info_idx = motif_info_mut['seq_idx']
    X_mut = X_in[motif_info_idx]
    if 0:
        #pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(motif_info_mut)
    
    for df_idx in range(0,seq_total):
        #print('Sequence index: %s' % df_idx)
        startA = motif_info_mut.loc[df_idx][4]
        stopA = startA + motif_A_len
        if model_pad != 'full':
            start_full = startA - model_pad
            stop_full = stopA + model_pad
        else:
            start_full = 0
            stop_full = maxL
        
        #if df_idx==0:
            #print(OH)
            #print(PWM)
            
        if motif_A_name == '13_AP1' or motif_A_name == 'AP1':# or motif_A_name == 'Ohler1': #correct for palindromic sequences without symmetric PWMs
            OH_F = X_mut[df_idx][startA-left_pad:stopA+right_pad]
            OH_R = X_mut[df_idx][startA-right_pad:stopA+left_pad]
            pwm_score_F = np.trace(np.dot(OH_F, PWM.T))
            PWM_R = pd.DataFrame(pwm.reverse_complement())
            pwm_score_R = np.trace(np.dot(OH_R, PWM_R.T))
            if pwm_score_F > pwm_score_R:
                pwm_scores.append(pwm_score_F)
            else:
                pwm_scores.append(pwm_score_R)
            
        else:
            OH = X_mut[df_idx][startA-left_pad:stopA+right_pad]
            pwm_scores.append(np.trace(np.dot(OH, PWM.T)))
            
        pwm_colors.append(cycle[core_muts-1])
        seq_info.append('%s_%s' % (core_muts, df_idx))

        
print('Max score index:', np.argmax(pwm_scores))
print('Min score index:', np.argmin(pwm_scores))
        
indexes, values = zip(*sorted(enumerate(pwm_scores), key=itemgetter(1), reverse=True))

pwm_colors = [pwm_colors[i] for i in indexes]
seq_info_ranked = [seq_info[i] for i in indexes]

if 0:
    for i in range(len(values)):
        print(i, seq_info_ranked[i])

barplot = plt.bar(np.arange(len(values)), values)
for b in range(len(indexes)):
    barplot[b].set_color(pwm_colors[b])
    

colors = {'0':cycle[-1], '1':cycle[0], '2':cycle[1]}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc='best')
plt.ylabel('PWM scores')
plt.xlabel('PWM rank')
#plt.show()

if save:
    np.save(os.path.join(saveDir, 'PWM_scores_%s_%s.npy' % (example, motif_A_name)), values)
    np.save(os.path.join(saveDir, 'PWM_colors_%s_%s.npy' % (example, motif_A_name)), pwm_colors)

plt.close()

# =============================================================================
# Sort attribution maps errors by PWM ranks and plot
# =============================================================================
color_ISM = '#377eb8' #blue
color_add = '#4daf4a' #green
if other == 'Saliency':
    color_other = '#ff7f00' #orange
elif other == 'DeepExplainer':
    color_other = '#e41a1c' #red
else:
    color_other = '#984ea3' #purple

add_errors_sort = [add_errors[i] for i in indexes]
ism_errors_sort = [ism_errors[i] for i in indexes]
other_errors_sort = [other_errors[i] for i in indexes]
if save:
    np.save(os.path.join(saveDir, 'Errors_add_%s_%s.npy' % (example, motif_A_name)), add_errors_sort)
    np.save(os.path.join(saveDir, 'Errors_ISM_%s_%s.npy' % (example, motif_A_name)), ism_errors_sort)
    np.save(os.path.join(saveDir, 'Errors_other_%s_%s.npy' % (example, motif_A_name)), other_errors_sort)
    
mwu_stat1, pval1 = stats.mannwhitneyu(add_errors_sort, ism_errors_sort, alternative='less')
print('p-values (add vs ism):', pval1)
mwu_stat2, pval2 = stats.mannwhitneyu(add_errors_sort, other_errors_sort, alternative='less')
print('p-values (add vs other):', pval2)
np.save(os.path.join(saveDir, 'PWM_pvalues_%s_%s.npy' % (example, motif_A_name)), np.array([pval1, pval2]))


if 0:
    fig, ax = plt.subplots(1,3, figsize=(15,3))
    
    barplot = ax[0].bar(np.arange(len(values)), ism_errors_sort)
    for b in range(len(indexes)):
        barplot[b].set_color(pwm_colors[b])
    ax[0].set_title('ISM')
    ax[0].set_ylabel('Attribution error')
    ax[0].set_xlabel('PWM rank')
    
    barplot = ax[1].bar(np.arange(len(values)), other_errors_sort)
    for b in range(len(indexes)):
        barplot[b].set_color(pwm_colors[b])
    ax[1].set_title(other)
    ax[1].set_xlabel('PWM rank')
    
    barplot = ax[2].bar(np.arange(len(values)), add_errors_sort)
    for b in range(len(indexes)):
        barplot[b].set_color(pwm_colors[b])
    ax[2].set_title('Additive')
    ax[2].set_xlabel('PWM rank')
    
    ax[0].sharex(ax[1])
    ax[1].sharey(ax[2])
    
    plt.tight_layout()
    plt.show()

if 0:    
    plt.plot(ism_errors_sort, c=color_ISM, label='ISM')
    plt.plot(other_errors_sort, c=color_other, label=other)
    plt.plot(add_errors_sort, c=color_add, label='Additive')
    plt.xlabel('PWM rank')
    plt.ylabel('Attribution error')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
if 0: #apply Savitzky-Golay filter : window size (odd); polynomial order
    width = 35
    poly = 6 #5
    ism_smooth = signal.savgol_filter(ism_errors_sort, width, poly)
    other_smooth = signal.savgol_filter(other_errors_sort, width, poly)
    add_smooth = signal.savgol_filter(add_errors_sort, width, poly)
    plt.plot(ism_errors_sort, c=color_ISM, alpha=0.2)
    plt.plot(ism_smooth, c=color_ISM, linestyle='-', label='ISM')
    plt.plot(other_errors_sort, c=color_other, alpha=0.2)
    plt.plot(other_smooth, c=color_other, linestyle='-', label=other)
    plt.plot(add_errors_sort, c=color_add, alpha=0.2)
    plt.plot(add_smooth, c=color_add, linestyle='-', label='Additive')
    plt.xlabel('PWM rank')
    plt.ylabel('Attribution error')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
if 0: #apply running mean
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    N = 25
    ism_smooth = running_mean(ism_errors_sort, N)
    other_smooth = running_mean(other_errors_sort, N)
    add_smooth = running_mean(add_errors_sort, N)
    plt.plot(ism_errors_sort, c=color_ISM, alpha=0.2)
    plt.plot(ism_smooth, c=color_ISM, linestyle='-', label='ISM')
    plt.plot(other_errors_sort, c=color_other, alpha=0.2)
    plt.plot(other_smooth, c=color_other, linestyle='-', label=other)
    plt.plot(add_errors_sort, c=color_add, alpha=0.2)
    plt.plot(add_smooth, c=color_add, linestyle='-', label='Additive')
    plt.xlabel('PWM rank')
    plt.ylabel('Attribution error')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if 1:
    from scipy.ndimage import uniform_filter1d
    N = 20#35
    ism_smooth = uniform_filter1d(ism_errors_sort, size=N)
    other_smooth = uniform_filter1d(other_errors_sort, size=N)
    add_smooth = uniform_filter1d(add_errors_sort, size=N)
    plt.plot(ism_errors_sort, c=color_ISM, alpha=0.2)
    plt.plot(ism_smooth, c=color_ISM, linestyle='-', label='ISM')
    plt.plot(other_errors_sort, c=color_other, alpha=0.2)
    plt.plot(other_smooth, c=color_other, linestyle='-', label=other)
    plt.plot(add_errors_sort, c=color_add, alpha=0.2)
    plt.plot(add_smooth, c=color_add, linestyle='-', label='Additive')
    plt.xlabel('PWM rank')
    plt.ylabel('Attribution error')
    #plt.legend(loc='best')
    plt.ylim(0,35)
    plt.tight_layout()
    plt.show()


