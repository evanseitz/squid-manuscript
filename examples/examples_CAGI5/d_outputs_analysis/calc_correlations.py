import os, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logomaker
import tfomics
from scipy import stats
import seaborn as sns

# use environment: 'gopher'

if 1:
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
    matplotlib.rc('text', usetex=True)
    plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{amssymb}",
        r"\usepackage{bold-extra}"])

pyDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(pyDir)
grandParentDir = os.path.dirname(parentDir)
greatGrandParentDir = os.path.dirname(grandParentDir)
sys.path.append(pyDir)
sys.path.append(parentDir)
sys.path.append(grandParentDir)
sys.path.append(greatGrandParentDir)

import squid.utils as squid_utils
import squid.figs_surrogate as squid_figs
alphabet = ['A','C','G','T']
alpha = 'dna'

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

df = pd.DataFrame(columns = ['model', 'difference', 'attribution']) #for final combined figure
df_idx = 0

attr_dirs = [
    os.path.join(parentDir, 'c_surrogate_outputs/model_ResidualBind32_Exp_single'), #100k_pad400
    os.path.join(parentDir, 'c_surrogate_outputs/model_Basenji32_GELU_single'), #100k_pad400
    os.path.join(parentDir, 'c_surrogate_outputs/model_ENFORMER_agnostic_DNASE'), #100k_pad400
    ]


model_names = [
    'ResidualBind-32\nSingle Exp, sum \nN=100k, pad=400',
    'Basenji-32\nSingle GELU, sum \nN=100k, pad=400',
    'ENFORMER\nagnostic, DNASE\nN=100k, pad=400',
    ]

loci = ['LDLR','ZFAND3','F9','SORT1','TERT-HEK293T', 
        'MSMB', 'IRF6', 'TERT-GBM', 'IRF4','PKLR',
        'HBB','HBG1','HNF4A','MYCrs6983267','GP1BB'
        ]

suite = ['GOPHER',
         'GOPHER',
         'ENFORMER',
         ]

outname = ['ResidualBind32', 
           'Basenji32', 
           'Enformer'
           ]


gauge = 'wildtype' #options: {None, 'wildtype', 'hierarchical'}
spearman = False

pvals_add_ism = [] #using additive nonlinear model
pvals_add_sal = [] #using additive nonlinear model
pvals_add_ism_L = [] #using additive linear model
pvals_add_sal_L = [] #using additive linear model
pearson_mean_sal = []
pearson_mean_ism = []
pearson_mean_add =[] #nonlinear
pearson_mean_add_L = [] #linear

for model_idx, model_name in enumerate(model_names):
    print('')
    print('Model name: %s' % model_name)

    if suite[model_idx] == 'GOPHER':
        maxL = 2048
    elif suite[model_idx] == 'ENFORMER':
        maxL = 393216

    sense = np.ones(15)
    # (arbitrary) sense flip required for loci modeled by linear (L) additive model
    if 0:
        sense_L_resbind = [1.,-1.,-1.,1.,1.,-1.,-1.,1.,-1.,1.,1.,1.,1.,1.,-1.]
        sense_L_basenji = [1.,1.,-1.,1.,-1.,1.,-1.,-1.,1.,1.,1.,-1.,-1.,1.,-1.]
        sense_L_enformer = [-1.,1.,-1.,1.,1.,-1.,-1.,1.,-1.,-1.,-1.,1.,-1.,1.,1.]
    else:
        sense_L_resbind = sense
        sense_L_basenji = sense
        sense_L_enformer = sense

    pearson_ism = []
    pearson_sal = []
    pearson_add = [] #nonlinear
    pearson_add_L = [] #linear
    if spearman is True:
        spearman_ism = []
        spearman_sal = []
        spearman_add = []
        spearman_add_L = []
    for idx, locus in enumerate(loci):
        cagiScores = pd.read_csv(os.path.join(parentDir,'a_model_assets/CAGI5_info/saturation_scores/combined_%s.tsv' % locus), header=6, delimiter='\t')
        locus_pos = cagiScores.iloc[0]['Pos'] #to keep track of gaps in cagiScores
        L = cagiScores.iloc[-1]['Pos'] - cagiScores.iloc[0]['Pos'] #length of each locus
        print('Idx: %s, Locus: %s, L: %s' % (idx, locus, L))

        # get one hot encoding (2048-bp region surrounding each 600-bp CAGI5 site)
        import h5py
        with h5py.File(os.path.join(parentDir, 'a_model_assets/CAGI5_onehots_centered_%s.h5' % maxL), 'r') as dataset:
            OHs_wt = np.array(dataset['reference']).astype(np.float32) #shape : (15, 2048, 4)
        OH_wt_full = OHs_wt[idx] #see 'centered_combined_cagi.bed for information on indexing
        centers = pd.read_csv(os.path.join(parentDir, 'a_model_assets/CAGI5_info/centered_combined_cagi.csv'), header=0, delimiter=',', index_col=0)
        center = centers.loc[centers['locus'] == locus]['center'][0] #the center of the CAGI locus

        loci_sites = pd.read_csv(os.path.join(parentDir,'b_recognition_sites/CAGI5-%s_positions.csv' % suite[model_idx]))
        loci_start = list(loci_sites.iloc[:]['locus_start'])

        OH_wt = OH_wt_full[loci_start[idx]:loci_start[idx]+L]

        #load deepnet model predictions:
        attr_ism = np.load(os.path.join(attr_dirs[model_idx], 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/attributions_ISM_single.npy' % (suite[model_idx],idx,idx)))
        attr_ism = attr_ism[loci_start[idx]:loci_start[idx]+L]
        attr_sal = np.load(os.path.join(attr_dirs[model_idx], 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/attributions_saliency.npy' % (suite[model_idx],idx,idx)))
        attr_sal = attr_sal[loci_start[idx]:loci_start[idx]+L]
        try:
            attr_add = np.array(pd.read_csv(os.path.join(attr_dirs[model_idx], 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/logo_additive.csv' % (suite[model_idx],idx,idx)), index_col=0))
        except OSError as e:
            attr_add = np.array(pd.read_csv(os.path.join(attr_dirs[model_idx], 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/mavenn_additive.csv' % (suite[model_idx],idx,idx)), index_col=0))
        attr_add = attr_add[loci_start[idx]:loci_start[idx]+L]*sense[idx]
        #attr_add_L = np.array(pd.read_csv(os.path.join(attr_dirs[model_idx]+'_linear', 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/mavenn_additive.csv' % (suite[model_idx],idx,idx)), index_col=0))
        attr_add_L = np.array(pd.read_csv(os.path.join(attr_dirs[model_idx], 'SQUID_CAGI5-%s_intra_mut0/rank%s_seq%s/ridge_additive.csv' % (suite[model_idx],idx,idx)), index_col=0))

        if model_idx == 0:
            attr_add_L = attr_add_L[loci_start[idx]:loci_start[idx]+L]*sense_L_resbind[idx]
        elif model_idx == 1:
            attr_add_L = attr_add_L[loci_start[idx]:loci_start[idx]+L]*sense_L_basenji[idx]
        elif model_idx == 2:
            attr_add_L = attr_add_L[loci_start[idx]:loci_start[idx]+L]*sense_L_enformer[idx]

        if gauge == 'wildtype':
            wt_argmax = np.argmax(OH_wt, axis=1)
            for l in range(L):
                wt_val = attr_sal[l,wt_argmax[l]]
                attr_sal[l,:] -= wt_val
            for l in range(L):
                wt_val = attr_add[l,wt_argmax[l]]
                attr_add[l,:] -= wt_val
            for l in range(L):
                wt_val = attr_add_L[l,wt_argmax[l]]
                attr_add_L[l,:] -= wt_val

        elif gauge == 'hierarchical':
            for l in range(L):
                col_mean = np.mean(attr_ism[l,:])
                attr_ism[l,:] -= col_mean
            for l in range(L):
                col_mean = np.mean(attr_sal[l,:])
                attr_sal[l,:] -= col_mean
            for l in range(L):
                col_mean = np.mean(attr_add[l,:])
                attr_add[l,:] -= col_mean
            for l in range(L):
                col_mean = np.mean(attr_add_L[l,:])
                attr_add_L[l,:] -= col_mean

        #collect activity map from experimental readout
        activity_df = np.zeros(shape=(L,4))
        OH_gaps = np.zeros(shape=(L,4)) #wild-type one hot encoding

        bp_idx = 0
        flat_act = []
        flat_ism = []
        flat_sal = []
        flat_add = []
        flat_add_L = []
        for bp in range(L):
            if (cagiScores.iloc[bp_idx]['Pos'] == locus_pos) and (cagiScores.iloc[bp_idx+1]['Pos'] == locus_pos) and (cagiScores.iloc[bp_idx+2]['Pos'] == locus_pos):
                if cagiScores.iloc[bp_idx]['Ref'] == 'A':
                    var_x = 1
                    var_y = 2
                    var_z = 3
                    OH_gaps[bp,0] = 1.
                elif cagiScores.iloc[bp_idx]['Ref'] == 'C':
                    var_x = 0
                    var_y = 2
                    var_z = 3
                    OH_gaps[bp,1] = 1.
                elif cagiScores.iloc[bp_idx]['Ref'] == 'G':
                    var_x = 0
                    var_y = 1
                    var_z = 3
                    OH_gaps[bp,2] = 1.
                elif cagiScores.iloc[bp_idx]['Ref'] == 'T':
                    var_x = 0
                    var_y = 1
                    var_z = 2
                    OH_gaps[bp,3] = 1.
                else:
                    print('Error at bp:', bp)
                
                if gauge == 'hierarchical':
                    activity_df[bp,var_x] = cagiScores.iloc[bp_idx]['Value']
                    bp_idx += 1
                    activity_df[bp,var_y] = cagiScores.iloc[bp_idx]['Value']
                    bp_idx += 1
                    activity_df[bp,var_z] = cagiScores.iloc[bp_idx]['Value']
                    bp_idx += 1
                    locus_pos += 1

                    col_mean = np.mean(activity_df[bp,:])
                    activity_df[bp,:] -= col_mean
                    flat_act.append(activity_df[bp,var_x])
                    flat_act.append(activity_df[bp,var_y])
                    flat_act.append(activity_df[bp,var_z])

                else: #original gauge for experimental data
                    activity_df[bp,var_x] = cagiScores.iloc[bp_idx]['Value']
                    flat_act.append(cagiScores.iloc[bp_idx]['Value'])
                    bp_idx += 1
                    activity_df[bp,var_y] = cagiScores.iloc[bp_idx]['Value']
                    flat_act.append(cagiScores.iloc[bp_idx]['Value'])
                    bp_idx += 1
                    activity_df[bp,var_z] = cagiScores.iloc[bp_idx]['Value']
                    flat_act.append(cagiScores.iloc[bp_idx]['Value'])
                    bp_idx += 1
                    locus_pos += 1

                flat_ism.append(attr_ism[bp,var_x])
                flat_ism.append(attr_ism[bp,var_y])
                flat_ism.append(attr_ism[bp,var_z])
                flat_sal.append(attr_sal[bp,var_x])
                flat_sal.append(attr_sal[bp,var_y])
                flat_sal.append(attr_sal[bp,var_z])
                flat_add.append(attr_add[bp,var_x])
                flat_add.append(attr_add[bp,var_y])
                flat_add.append(attr_add[bp,var_z])
                flat_add_L.append(attr_add_L[bp,var_x])
                flat_add_L.append(attr_add_L[bp,var_y])
                flat_add_L.append(attr_add_L[bp,var_z])

            else: #skip over gaps in cagiScores file
                if 0:
                    print('Incomplete BP:', locus_pos)
                attr_ism[bp,:] = 0
                attr_sal[bp,:] = 0
                attr_add[bp] = 0
                attr_add_L[bp] = 0
                locus_pos += 1
                OH_wt[bp,:] = 0 #for sanity check only
                continue

        if 0: #view OH as sequence
            OH_a = squid_utils.oh2seq(OH_wt, alphabet=['A','C','G','T']) #GRCh37/hg19
            OH_b = squid_utils.oh2seq(OH_gaps, alphabet=['A','C','G','T'])
            print(len(OH_a), len(OH_b))
            if 1:
                def hammingDist(str1, str2):
                    i = 0
                    count = 0
                    while(i < len(str1)):
                        if(str1[i] != str2[i]):
                            count += 1
                        i += 1
                    return count
                hamming = hammingDist(OH_a, OH_b)
                print('Hamming:', hamming)
                if hamming != 0:
                    print(OH_a[0:15])
                    print(OH_b[0:15])

        if 0:
            fig, axs = plt.subplots(1,4, figsize=(10,5))
            axs[0].scatter(flat_act, flat_ism, s=5, c='k', alpha=.2)
            axs[1].scatter(flat_act, flat_sal, s=5, c='k', alpha=.2)
            axs[2].scatter(flat_act, flat_add_L, s=5, c='k', alpha=.2)
            axs[3].scatter(flat_act, flat_add, s=5, c='k', alpha=.2)
            axs[0].set_ylabel('ISM')
            axs[1].set_ylabel('Saliency')
            axs[2].set_ylabel('Additive (Linear)')
            axs[3].set_ylabel('Additive (Nonlinear)')
            axs[0].set_xlabel('CAGI5')
            axs[1].set_xlabel('CAGI5')
            axs[2].set_xlabel('CAGI5')
            axs[3].set_xlabel('CAGI5')
            plt.tight_layout()
            plt.show()

        pearson_ism.append(stats.pearsonr(flat_act, flat_ism)[0])
        pearson_sal.append(stats.pearsonr(flat_act, flat_sal)[0])
        pearson_add.append(stats.pearsonr(flat_act, flat_add)[0])
        pearson_add_L.append(stats.pearsonr(flat_act, flat_add_L)[0])

        if spearman is True:
            spearman_ism.append(stats.spearmanr(flat_act, b=flat_ism, alternative='two-sided')[0])
            spearman_sal.append(stats.spearmanr(flat_act, b=flat_sal, alternative='two-sided')[0])
            spearman_add.append(stats.spearmanr(flat_act, b=flat_add, alternative='two-sided')[0])
            spearman_add_L.append(stats.spearmanr(flat_act, b=flat_add_L, alternative='two-sided')[0])

        if 0:
            print('Individual locus scores:')
            print('Pearson ISM:', pearson_ism)
            print('Pearson Saliency:', pearson_sal)
            print('Pearson Additive (linear):', pearson_add_L)
            print('Pearson Additive (nonlinear):', pearson_add)
            print('Add-ISM (linear):', np.array(pearson_add_L)-np.array(pearson_ism))
            print('Add-ISM (nonlinear):', np.array(pearson_add)-np.array(pearson_ism))

            print('')
            if spearman is True:
                print('Spearman ISM:', spearman_ism)
                print('Spearman Saliency:', spearman_sal)
                print('Spearman Additive (linear):', spearman_add_L)
                print('Spearman Additive (nonlinear):', spearman_add)

        # plot attribution maps for comparison
        if 0:
            if locus == 'IRF4':
                fig, axs = plt.subplots(4, 1, figsize=([10,4*0.7]))
                axs[0].set_title(locus, fontsize=12)

                activity_logo = squid_utils.l2_norm_to_df(OH_gaps, activity_df, alphabet=alphabet, alpha=alpha)
                logomaker.Logo(df=activity_logo,
                                ax=axs[0],
                                fade_below=.5,
                                shade_below=.5,
                                width=.9,
                                center_values=False,
                                color_scheme='classic',
                                font_name='Arial Rounded MT Bold')
                axs[0].set_ylabel('CAGI5', fontsize=8)
                axs[0].get_xaxis().set_ticks([])
                axs[0].get_yaxis().set_ticks([])

                ism_logo = squid_utils.l2_norm_to_df(OH_gaps, attr_ism, alphabet=alphabet, alpha=alpha)
                logomaker.Logo(df=ism_logo,
                                ax=axs[1],
                                fade_below=.5,
                                shade_below=.5,
                                width=.9,
                                center_values=False,
                                color_scheme='classic',
                                font_name='Arial Rounded MT Bold')
                axs[1].set_ylabel('ISM', fontsize=8)
                axs[1].get_xaxis().set_ticks([])
                axs[1].get_yaxis().set_ticks([])

                if gauge == None or gauge == 'hierarchical':
                    sal_logo = tfomics.impress.grad_times_input_to_df(np.expand_dims(OH_gaps, 0), attr_sal)
                    center = True
                elif gauge == 'wildtype':
                    sal_logo = squid_utils.l2_norm_to_df(OH_gaps, attr_sal, alphabet=alphabet, alpha=alpha)
                    sal_logo = squid_utils.arr2pd(attr_sal, alphabet)
                    center = False

                logomaker.Logo(df=sal_logo,
                                ax=axs[2],
                                fade_below=.5,
                                shade_below=.5,
                                width=.9,
                                center_values=center,
                                color_scheme='classic',
                                font_name='Arial Rounded MT Bold')
                axs[2].set_ylabel('Saliency', fontsize=8)
                axs[2].get_xaxis().set_ticks([])
                axs[2].get_yaxis().set_ticks([])

                '''if gauge == 'wildtype':
                    add_logo = squid_utils.l2_norm_to_df(OH_gaps, attr_add, alphabet=alphabet, alpha=alpha)
                    add_logo = squid_utils.arr2pd(attr_add, alphabet)
                    center = False
                else:
                    add_logo = squid_utils.arr2pd(attr_add, alphabet)
                    center = True'''

                logomaker.Logo(df= squid_utils.arr2pd(attr_add, alphabet),
                                ax=axs[3],
                                fade_below=.5,
                                shade_below=.5,
                                width=.9,
                                center_values=True,
                                color_scheme='classic',
                                font_name='Arial Rounded MT Bold')
                axs[3].set_ylabel('Additive (nonlinear)', fontsize=8)
                axs[3].get_yaxis().set_ticks([])

                axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))

                plt.tight_layout()
                plt.savefig('/Users/evanseitz/Documents/%s_compare_maps_%s_model%s.png' % (idx, locus, model_idx), facecolor='w', dpi=200)
                plt.show()

        # plot attribution matrices
        if 0:
            if locus == 'LDLR':
                if 0:
                    view_map = attr_ism
                    label = 'ISM'
                elif 0:
                    view_map = attr_add
                    label = 'ADD'
                elif 0:
                    view_map = attr_sal
                    label = 'SAL'
                else:
                    view_map = activity_df
                    label = 'GT'

                fig, ax = plt.subplots(1,1, figsize=(50,5))
                norm = matplotlib.colors.TwoSlopeNorm(vmin=view_map.min(), vcenter=0, vmax=view_map.max())
                im = ax.pcolormesh(view_map.T, linewidth=0, cmap='bwr', norm=norm)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                B = ['A', 'C', 'G', 'T']
                ax.set_yticks([0.5, 1.5, 2.5, 3.5])
                ax.set_yticklabels(B, fontsize=12)
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig('/Users/evanseitz/Documents/%s_compare_matrix_%s_%s_model%s.png' % (idx, locus, label, model_idx), facecolor='w', dpi=200)
                plt.show()

    if 1:
        color_ISM = '#377eb8' #blue
        color_sal = '#ff7f00' #orange
        color_add_L = '#f781bf' #pink
        color_add = '#4daf4a' #blue
        width = 0.2 #bar width
        ind = np.arange(len(loci))
        if spearman is True:
            fig, axs = plt.subplots(2, 1, figsize=(15,6))
        else:
            fig, axs = plt.subplots(figsize=(15,3))
        if spearman is True:
            ax = axs[0]
        else:
            ax = axs

        ax.set_ylabel('Pearson correlation', labelpad=8)

        ax.bar(ind-(width+(width/2)), pearson_sal, width, label='Saliency', color=color_sal)
        ax.bar(ind-(width/2), pearson_ism, width, label='ISM', color=color_ISM)
        ax.bar(ind+(width/2), pearson_add_L, width, label='Additive (linear)', color=color_add_L)
        ax.bar(ind+(width+(width/2)), pearson_add, width, label='Additive (nonlinear)', color=color_add)
        ax.set_xticks(np.arange(0, len(loci)+1, 1.0))
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for l in range(len(loci)):
            if loci[l] == 'MYCrs6983267':
                labels[l] = 'MYC'
            elif loci[l] == 'TERT-HEK293T':
                labels[l] = 'TERT\n(HEK293)'
            elif loci[l] == 'TERT-GBM':
                labels[l] = 'TERT\n(GBM)'
            else:
                labels[l] = loci[l]
        ax.set_xticklabels(labels)

        '''if spearman is True:
            ax = axs[1]
            ax.set_ylabel('Spearman correlation', labelpad=8)
            ax.bar(ind-width, spearman_sal, width, label='Saliency', color=color_sal)
            ax.bar(ind, spearman_ism, width, label='ISM', color=color_ISM)
            ax.bar(ind+width, spearman_add, width, label='Additive', color=color_add)  
            ax.set_xticks(np.arange(0, len(loci)+1, 1.0))
            labels = [item.get_text() for item in ax.get_xticklabels()]
            for l in range(len(loci)):
                if loci[l] == 'MYCrs6983267':
                    labels[l] = 'MYC'
                elif loci[l] == 'TERT-HEK293T':
                    labels[l] = 'TERT\n(HEK293)'
                elif loci[l] == 'TERT-GBM':
                    labels[l] = 'TERT\n(GBM)'
                else:
                    labels[l] = loci[l]
            ax.set_xticklabels(labels)'''

        plt.legend(loc='best', prop={'size': 8})
        plt.tight_layout()
        if 1:
            plt.savefig(os.path.join(pyDir,'pearson_barplots_%s.pdf' % (outname[model_idx])), facecolor='w', dpi=200)
        #plt.tight_layout()
        plt.show()

        pearson_mean_sal.append(np.mean(np.array(pearson_sal)))
        pearson_mean_ism.append(np.mean(np.array(pearson_ism)))
        pearson_mean_add.append(np.mean(np.array(pearson_add)))
        pearson_mean_add_L.append(np.mean(np.array(pearson_add_L)))

        print('mean(Pearson | Saliency):', pearson_mean_sal[-1])
        print('mean(Pearson | ISM):', pearson_mean_ism[-1])
        print('mean(Pearson | Additive (linear)):', pearson_mean_add_L[-1])
        print('mean(Pearson | Additive (nonlinear)):', pearson_mean_add[-1])

        '''if spearman:
            print('mean(Spearman | Saliency):',np.mean(np.array(spearman_sal)))
            print('mean(Spearman | ISM):',np.mean(np.array(spearman_ism)))
            print('mean(Spearman | Additive):',np.mean(np.array(spearman_add)))'''

    if 1:
        ADD_diff_ISM_L = []
        ADD_diff_SAL_L = []
        ADD_diff_ISM = []
        ADD_diff_SAL = []
        ADD_diff_ADD_L = []
        for i in range(len(loci)):
            ADD_diff_ISM_L.append(pearson_add_L[i] - pearson_ism[i])
            ADD_diff_SAL_L.append(pearson_add_L[i] - pearson_sal[i])
            ADD_diff_ISM.append(pearson_add[i] - pearson_ism[i])
            ADD_diff_SAL.append(pearson_add[i] - pearson_sal[i])
            ADD_diff_ADD_L.append(pearson_add[i] - pearson_add_L[i])
        print('')
        print('Wilcoxon | Pearson (linear):')
        pval1 = stats.wilcoxon(ADD_diff_ISM_L, alternative='greater')[1]
        pval2 = stats.wilcoxon(ADD_diff_SAL_L, alternative='greater')[1]
        print('Add–ISM:',pval1)
        print('Add-Sal:',pval2)
        pvals_add_ism_L.append(pval1)
        pvals_add_sal_L.append(pval2)
        print('')
        print('Wilcoxon | Pearson (nonlinear):')
        pval1 = stats.wilcoxon(ADD_diff_ISM, alternative='greater')[1]
        pval2 = stats.wilcoxon(ADD_diff_SAL, alternative='greater')[1]
        print('Add–ISM:',pval1)
        print('Add-Sal:',pval2)
        pvals_add_ism.append(pval1)
        pvals_add_sal.append(pval2)
        print('')
        print('Wilcoxon | Pearson (linear vs. nonlinear):')
        pval3 = stats.wilcoxon(ADD_diff_ADD_L, alternative='greater')[1]
        print('Add_NL-Add_L:',pval3)
        '''if spearman:
            ADD_diff_ISM = []
            ADD_diff_SAL = []
            for i in range(len(loci)):
                ADD_diff_ISM.append(spearman_add[i] - spearman_ism[i])
                ADD_diff_SAL.append(spearman_add[i] - spearman_sal[i])
            print('Wilcoxon | Spearman:')
            pval1 = stats.wilcoxon(ADD_diff_ISM, alternative='greater')[1]
            pval2 = stats.wilcoxon(ADD_diff_SAL, alternative='greater')[1]
            print('Add–ISM:',pval1)
            print('Add-Sal:',pval2)
            print('')
            print('')'''

        for i in range(len(loci)*2):
            if i < len(loci):
                df.at[df_idx, 'model'] = model_name
                df.at[df_idx, 'difference'] = ADD_diff_ISM[i]
                df.at[df_idx, 'attribution'] = 'Additive – ISM'
                df_idx += 1
            else:
                df.at[df_idx, 'model'] = model_name
                df.at[df_idx, 'difference'] = ADD_diff_SAL[i-len(loci)]
                df.at[df_idx, 'attribution'] = 'Additive – Saliency'
                df_idx += 1



if 0:
    #print(pearson_mean_ism)
    #print(pearson_mean_add)
    fig, ax = plt.subplots()
    fig.suptitle('Average Pearson correlation across models')
    for i in range(len(pearson_mean_add)):
        if pvals_add_ism[i] < 0.001:
            ax.scatter(pearson_mean_ism[i], pearson_mean_add[i], c='#d7191c', label='$P$ < 0.001', edgecolors='k')
        elif pvals_add_ism[i] < 0.01:
            ax.scatter(pearson_mean_ism[i], pearson_mean_add[i], c='#fdae61', label='$P$ < 0.01', edgecolors='k')
        elif pvals_add_ism[i] < 0.05:
            ax.scatter(pearson_mean_ism[i], pearson_mean_add[i], c='#abdda4', label='$P$ < 0.05', edgecolors='k',)
        else:
            ax.scatter(pearson_mean_ism[i], pearson_mean_add[i], c='#2b83ba', label='$ns$', edgecolors='k')
    ax.plot([0, 1], [0, 1],  '--', c='k', transform=ax.transAxes)
    mean_min = np.min([np.min(pearson_mean_ism),np.min(pearson_mean_add)])
    mean_max = np.max([np.max(pearson_mean_ism),np.max(pearson_mean_add)])
    ax.set_xlim(mean_min-.005,mean_max+.005)
    ax.set_ylim(mean_min-.005,mean_max+.005)
    ax.set_xlabel('ISM')
    ax.set_ylabel('Additive')
    ax.set_aspect('equal', 'box')
    #plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    print('MWU p-value:')
    print(stats.mannwhitneyu(pearson_mean_add, pearson_mean_ism, alternative='greater'))


# final figure combining the Pearson differences between attribution maps for every model
if 0:
    fig, ax = plt.subplots(figsize=(10, 9)) #(10, 9) for 3 rows

    df = df.astype({'difference': float})
    #(or sns.striplot)
    swarm = sns.swarmplot(data=df, ax=ax, x="difference", y="model", hue="attribution", dodge=True, size=10, #5
                    palette={"Additive – ISM": color_ISM, "Additive – Saliency": color_sal})
    swarm.legend_.set_title(None)

    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 3},
                #medianprops={'visible': True, 'lw': 3},
                medianprops={'visible': False},
                whiskerprops={'visible': False},
                zorder=10,
                x="difference",
                y="model",
                data=df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                hue="attribution",
                dodge=True,
                ax=swarm)#,
                #width=1.1)

    swarm.legend_.remove()
    swarm.set(ylabel=None)
    ax.axvline(0, c='lightgray')
    sns.set(font_scale=4)
    #sns.set(rc={'figure.figsize':(2,4)}) #(width, height)
    #swarm.legend()
    plt.tight_layout()
    plt.show()

print('All Wilcoxon p-values (ADD-ISM; ADD-SAL):')
for pval in range(len(model_names)):
    print('  %s: %s; %s (linear)' % (pval, pvals_add_ism_L[pval], pvals_add_sal_L[pval]))
    print('  %s: %s; %s (nonlinear)' % (pval, pvals_add_ism[pval], pvals_add_sal[pval]))


        


