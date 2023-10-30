import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import pandas as pd
import logomaker
import scipy
from scipy import stats
from Bio import motifs #pip install biopython
import h5py

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils

np.random.seed(0)

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])

#model = 'GOPHER'
model = 'BPNet'

if model == 'GOPHER':
    model_type = 'ResidualBind32_ReLU_single'
    motif_name = '13_AP1'
    rank = 1
    seq = 641
    use_mut = 0
elif model == 'BPNet':
    model_type = 'BPNet_OSKN'
    motif_name = 'Nanog-long'
    rank = 12
    seq = 8176
    use_mut = 2

fig_pad = 75

# load PWM via path to position frequency matrix:
if model == 'GOPHER':
    pfm_fname = os.path.join(parentDir,'examples_%s/b_recognition_sites/PWMs/known_PWMs/ap1__MA0476.1.pfm' % model)
elif model == 'BPNet':
    pfm_fname = os.path.join(parentDir,'examples_%s/b_recognition_sites/PWMs/known_PWMs/nanog__HUMAN.H11MO.1.B.pcm' % model)

PFM = motifs.read(open(pfm_fname), 'jaspar') #position frequency matrix
PPM = PFM.counts.normalize(pseudocounts=0.5) #position probability matrix
pwm = PPM.log_odds() #position specific scoring matrix; i.e., position weight matrix (PWM)
pwm_rc = pwm.reverse_complement()
PWM = pd.DataFrame(pwm)
PWM_rc = pd.DataFrame(pwm_rc)

alphabet = ['A','C','G','T']
alpha = 'dna'
motif_len = PWM.shape[0]
wtFolder_NL = 'SQUID_%s_intra_mut%s' % (motif_name, use_mut)

color_ISM = '#377eb8' #blue
color_sal = '#ff7f00' #orange
color_dE = '#e41a1c' #red
color_dL = '#984ea3' #purple
color_add = '#4daf4a' #green

ISM_dir = os.path.join(parentDir, 'examples_%s/c_surrogate_outputs/model_%s/%s/rank%s_seq%s' % (model, model_type, wtFolder_NL, rank, seq))
ISM = np.load(os.path.join(ISM_dir, 'attributions_ISM_single.npy'))
ISM = ISM[:-1,:]
ISM = squid_utils.arr2pd(ISM, alphabet)

# retrieve wild-type sequence:
motif_info = pd.read_csv(os.path.join(parentDir,'examples_%s/b_recognition_sites/model_%s/%s_positions.csv' % (model, model_type, motif_name)))
motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
motif_info = motif_info.loc[motif_info['motif_mutations'] == use_mut]
motif_info.reset_index(drop=True, inplace=True)
motif_info_idx = motif_info['seq_idx']

if 1:
    start = motif_info['motif_start'][rank]
    stop = start + motif_len
else:
    start, stop = 470, 510

if model == 'GOPHER':
    with h5py.File(os.path.join(parentDir, 'examples_GOPHER/a_model_assets/cell_line_%s.h5' % 13), 'r') as dataset:
        X_in = np.array(dataset['X']).astype(np.float32)
elif model == 'BPNet':
    with h5py.File(os.path.join(parentDir, 'examples_BPNet/a_model_assets/bpnet_seqs_chr1-8-9.h5'), 'r') as dataset:
        X_in = np.array(dataset['X']).astype(np.float32)

X_in = X_in[motif_info_idx] #shape=(2048,4)
OH_wt = X_in[rank]
WT = squid_utils.oh2seq(OH_wt,alphabet)#[start:stop]
WT_list = [*WT]

fig, axs = plt.subplots(3,1,figsize=(30,5)) #,15 or 5)

if 1: #visualize PWM
    blank = np.zeros(shape=(len(WT), 4))
    blank[start:start+motif_len,:] = PWM
    pwm_long = squid_utils.arr2pd(blank, alphabet)
    logo = logomaker.Logo(pwm_long, ax=axs[0], center_values=True,
                        font_name='Arial Rounded MT Bold',
                        fade_below=.5, shade_below=.5)
    axs[0].set_xlim(start-fig_pad, stop+fig_pad)
    axs[0].tick_params(axis='x', which='major', labelsize=24)
    if 0:
        axs[0].tick_params(axis='y', which='major', labelsize=24)
    else:
        axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)


if 1: #visualize ISM attribution map
    logo = logomaker.Logo(ISM[start-100:stop+100], ax=axs[1], center_values=True,
                        font_name='Arial Rounded MT Bold',
                        #fade_below=.5, shade_below=.25,
                        fade_below=0, shade_below=0,
                        color_scheme='darkgray')
    logo.style_glyphs_in_sequence(sequence=WT[start-100:stop+100], color=color_ISM)

    #axs[1].set_xlim(xlims[0],xlims[1])
    axs[1].tick_params(axis='x', which='major', labelsize=24)
    if 0:
        axs[1].tick_params(axis='y', which='major', labelsize=24)
    else:
        axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_xlim(start-fig_pad, stop+fig_pad)

    #axs[1].axvline(1081, c='C3', linewidth=2.5, zorder=-50)
    #axs[1].axvline(1092, c='C2', linewidth=2.5, zorder=-50)
    #axs[1].axvline(1112, c='C1', linewidth=2.5, zorder=-50)
    #axs[1].axvline(1183, c='C6', linewidth=2.5, zorder=-50)
    #axs[1].axvspan(1081, 1081+motif_len, alpha=.05, color='C3', zorder=-60)
    #axs[1].axvspan(1092, 1092+motif_len, alpha=.05, color='C2', zorder=-60)
    #axs[1].axvspan(1112, 1112+motif_len, alpha=.05, color='C1', zorder=-60)
    #axs[1].axvspan(1183, 1183+motif_len, alpha=.05, color='C6', zorder=-60)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].axhline(0, c='k', alpha=1)


if 1: #convolve PWM and OH
    idx = []
    convol = []

    for t in range(-100,100):
        pwm_scan = np.trace(np.dot(PWM, OH_wt[start-t:stop-t].T))

        if motif_name == '13_AP1':
            pwm_rc_scan = np.trace(np.dot(PWM_rc, OH_wt[start-t:stop-t].T))

            pwm_scan_max = np.amax([pwm_scan, pwm_rc_scan])
            if pwm_scan_max > 0.:
                convol.append(pwm_scan_max)
            else:
                convol.append(0.)
        else:
            if pwm_scan > 0.:
                convol.append(pwm_scan)
            else:
                convol.append(0.)
        idx.append(start-t)

    if len(convol) > 0:
        axs[2].plot(idx, convol, c='k', label='PWM')
        axs[2].fill_between(x=idx, y1=convol, color= "lightgray", alpha= 0.2)

        #axs[2].axvline(1081, c='C3', linewidth=2.5, zorder=-10)
        #axs[2].axvline(1092, c='C2', linewidth=2.5, zorder=-10)
        #axs[2].axvline(1112, c='C1', linewidth=2.5, zorder=-10)
        #axs[2].axvline(1183, c='C6', linewidth=2.5, zorder=-10)
        #axs[2].set_xlim(xlims[0],xlims[1])
        #axs[2].set_xlim(xlims[0],xlims[1])
        axs[2].tick_params(axis='x', which='major', labelsize=24)
        if 0:
            axs[2].tick_params(axis='y', which='major', labelsize=24)
        else:
            axs[2].set_yticks([])
        axs[2].set_xticks([])
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        axs[2].spines['left'].set_visible(False)
        axs[2].axhline(0, c='k', alpha=0.5)

    axs[2].set_xticks(range(0,len(WT)))
    axs[2].set_xticklabels(WT_list, fontsize=16)
    axs[2].set_xlim(start-fig_pad, stop+fig_pad)


plt.tight_layout()
if 1: #save files to local
    plt.savefig(os.path.join(pyDir,'convolution.pdf'), facecolor='w', dpi=200)
plt.show()