import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import pandas as pd
import logomaker
import scipy
from scipy import stats
import mavenn

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
sys.path.append(grandParentDir)
import squid.utils as squid_utils
import squid.figs_surrogate as squid_figs_surrogate

# environment: 'mavenn_citra'

np.random.seed(0)

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])
    
if 0:
    gauge = 'hierarchical'
else:
    gauge = 'default'

alphabet = ['A','C','G','T']
alpha = 'dna'

if 0:
    linearity = 'GE'
    dataDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_Exp_single/SQUID_13_AP1_N_13_AP1_N_inter_mut0/rank13_seq786_dist19')
    sense = 1.
    mavenn_model = mavenn.load(os.path.join(dataDir, 'mavenn_model'))
else:
    linearity = 'linear'
    dataDir = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_Exp_single/SQUID_13_AP1_N_13_AP1_N_inter_mut0/fig_rank13_seq786_dist19_PW_L')
    sense = 1.#-1.
    mavenn_model = mavenn.load(os.path.join(dataDir, 'mavenn_model_linear'))


theta_dict = mavenn_model.get_theta(gauge='empirical')
theta_lclc = theta_dict['theta_lclc']
theta_lclc[np.isnan(theta_lclc)] = 0
theta_lc = theta_dict['theta_lc'] #shape=(45, 4, 45, 4)
theta_lc[np.isnan(theta_lc)] = 0

theta_lclc *= sense
theta_lc *= sense


# fix gauge:
if gauge == 'hierarchical':
    theta_lc = squid_utils.fix_gauge(np.array(theta_lc), gauge='hierarchical', wt=None)

    for l1 in range(theta_lclc.shape[0]): #hierarchical gauge (pairwise)
        for l2 in range(theta_lclc.shape[0]):
            box_mean = np.mean(theta_lclc[l1,:,l2,:])
            theta_lclc[l1,:,l2,:] -= box_mean


# ensure consistent scaling between additive and pairwise colorbars
if linearity == 'GE':
    print('ADD min/max: %s, %s' % (np.amin(theta_lc), np.amax(theta_lc)))
    print('PW min/max: %s, %s' % (np.amin(theta_lclc), np.amax(theta_lclc)))
    theta_max = [abs(np.amin(theta_lc)), abs(np.amin(theta_lclc)), abs(np.amax(theta_lc)), abs(np.amax(theta_lclc))]
    theta_limit = np.amax(theta_max)
else: #compute linearity='GE' and fill in below
    theta_limit = 3.32


if 1: #plot additive logo
    fig, ax = plt.subplots(figsize=[10.5,1])
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

    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'ADD_locus_logo_%s.pdf' % linearity), facecolor='w', dpi=200)
        plt.show()
    else:
        plt.close()


if 1: #plot additive matrix
    divnorm=colors.TwoSlopeNorm(vmin=-1.*theta_limit, vcenter=0., vmax=theta_limit)

    fig, ax = plt.subplots(figsize=[10.5,1])
    im = plt.pcolormesh(theta_lc.T,
                        norm=divnorm,
                        edgecolors='k',
                        linewidth=.35,
                        cmap='seismic',
                        color='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3.0%', pad=0.15)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    B = ['A', 'C', 'G', 'T']
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(B, fontsize=12)

    plt.colorbar(im, cmap='seismic', cax=cax)
    plt.clim(-1.*theta_limit, theta_limit)

    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'ADD_locus_matrix_%s.pdf' % linearity), facecolor='w', dpi=200)
        plt.show()
    else:
        plt.close()


if 1: #plot pairwise matrix
    fig, ax = plt.subplots(figsize=[10,5])
    ax, cb = mavenn.heatmap_pairwise(values=theta_lclc,
                                    alphabet=alphabet,
                                    ax=ax,
                                    gpmap_type='pairwise',
                                    cmap_size='2%',
                                    show_alphabet=False,
                                    cmap='seismic',
                                    cmap_pad=.1,
                                    show_seplines=True,            
                                    sepline_kwargs = {'color': 'k',
                                                        'linestyle': '-',
                                                        'linewidth': .5,
                                                        'color':'gray'})
    start, stop, pad = 1071, 1116, 0
    ax.xaxis.set_ticks(np.arange(0,(stop+pad)-(start-pad),2))
    ax.set_xticklabels(np.arange((start-pad),(stop+pad),2))  
    cb.set_label(r'Pairwise Effect',
                    labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')

    plt.cm.ScalarMappable.set_clim(cb, vmin=-1.*theta_limit, vmax=theta_limit)

    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'PW_locus_matrix_%s.pdf' % linearity), facecolor='w', dpi=200)
        plt.show()
    else:
        plt.close()