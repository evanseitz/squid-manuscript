#env: use mavenn_citra

import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mavenn
import logomaker
import itertools
from scipy import stats


pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandparentDir = os.path.dirname(parentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandparentDir)
import squid.utils as squid_utils

# User parameters:
N = 100000 #total number of sequences
motifA_name = 'Oct4' #e.g., {'Oct4', 'Sox2', 'Klf4', 'Nanog'}
start, stop = 480, 530
alphabet = ['A','C','G','T']

dataDir = os.path.join(parentDir, 'global/outputs/%s' % (motifA_name))
logoDir = os.path.join(parentDir, 'global/outputs/%s' % (motifA_name))
if not os.path.exists(logoDir):
    os.mkdir(logoDir)


if 1: #analyze additive matrices
    fname = '%s_N%s_GE_additive' % (motifA_name, N)
    model = mavenn.load(os.path.join(dataDir, fname))
    theta_dict = model.get_theta(gauge='empirical')
    theta_dict.keys()
    theta_lc = theta_dict['theta_lc']*-1.

    if 1:
        fig, ax = plt.subplots(figsize=[10.5,2]) #[15,2]
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        logomaker.Logo(df=squid_utils.arr2pd(theta_lc, alphabet),
                        ax=ax,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold')
        ax.set_xlim(475+5,525+5)
        plt.tight_layout()
        plt.savefig(os.path.join(logoDir,'%s_N%s_logo_clean.png' % (motifA_name, N)), facecolor='w', dpi=600)
        #plt.show()
        plt.close()