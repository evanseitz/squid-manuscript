import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import mavenn
import seaborn as sns
from scipy.spatial import distance
import joblib

# environment: e.g., 'source activate mavenn_citra' – python 3.10.8; pandas 1.4.2 (need correct version of pandas used to originally create the mave dataframe)
    # conda create -n mavenn_citra python pandas=1.4.2
    # pip install mavenn
    # pip install mavenn --upgrade
    # pip install seaborn

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
sys.path.append(grandParentDir)
import squid.utils as squid_utils

'''model_name = 'model_ResidualBind32_ReLU_single'
motif_A_name = '13_AP1'
motif_info = pd.read_csv(os.path.join(parentDir,'examples_GOPHER/b_recognition_sites/%s/%s_positions.csv' % (model_name, motif_A_name)))
motif_info = motif_info.sort_values(by = ['motif_rank'], ascending = [False])
print(motif_info)'''

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
color_palette = ['#f781bf', '#4daf4a', '#4b4b4b', '#AAFF00']
sns.set_palette(palette=color_palette, n_colors=4)
#sns.color_palette("rocket")

alphabet = ['A','C','G','T']

linear = 'ridge' #options: {ridge, linear}

if linear == 'ridge':
    folder_L = 'SQUID_13_AP1_intra_mut0'#_ridge'
    dir_L = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank28_seq3783' % folder_L)
    mave_L = pd.read_pickle(os.path.join(dir_L, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
    model_L = joblib.load(os.path.join(dir_L,'ridge_model.pkl'))
    X = mave_L['x']
    Y = np.array(mave_L['y'])
    # calculate yhats:
    X_OH = np.zeros(shape=(mave_L['x'].shape[0], len(mave_L['x'][0]), len(alphabet)))
    for i in range(mave_L['x'].shape[0]):
        X_OH[i,:,:] = squid_utils.seq2oh(mave_L['x'][i], alphabet)
    X_OH = X_OH.reshape(X_OH.shape[0], -1)
    yhat_test_L = model_L.predict(X_OH)
    y_test_L = Y

elif linear == 'linear':
    folder_L = 'SQUID_13_AP1_intra_mut0_linear'
    dir_L = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank28_seq3783' % folder_L)
    model_L = mavenn.load(os.path.join(dir_L, 'mavenn_model'))
    mave_L = pd.read_pickle(os.path.join(dir_L, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
    trainval_df_L, test_df_L = mavenn.split_dataset(mave_L)
    y_test_L = test_df_L['y']
    phi_test_L = model_L.x_to_phi(test_df_L['x'])
    yhat_test_L = model_L.x_to_yhat(test_df_L['x'])

folder_NL = 'SQUID_13_AP1_intra_mut0'
dir_NL = os.path.join(parentDir, 'examples_GOPHER/c_surrogate_outputs/model_ResidualBind32_ReLU_single/%s/rank28_seq3783' % folder_NL)
model_NL = mavenn.load(os.path.join(dir_NL, 'mavenn_model'))
mave_NL = pd.read_pickle(os.path.join(dir_NL, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
trainval_df_NL, test_df_NL = mavenn.split_dataset(mave_NL)

y_test_NL = test_df_NL['y']
phi_test_NL = model_NL.x_to_phi(test_df_NL['x'])
yhat_test_NL = model_NL.x_to_yhat(test_df_NL['x'])

wt = 'TGAGTCA'
ism = pd.read_csv(os.path.join(dir_L,'attributions_ISM_single.csv'), index_col=0)
ism = ism[984:984+200]
ism[ism != 0] = 1.
wt_full = squid_utils.oh2seq(np.array(ism), ['A','C','G','T'])
hamming_df = pd.DataFrame(columns = ['y', 'phi', 'h'], index=range(len(y_test_NL)))

width = 0 #extent of flanking region around wild-type core sequence (TGAGTCA)
min_d = np.inf
min_d_idx = 0
for i, x in enumerate(test_df_NL['x']):
    d = round(distance.hamming(list(wt_full[100-width:107+width]), list(x[100-width:107+width])) * len(wt_full[100-width:107+width]))
    hamming_df.at[i, 'y'] = y_test_NL[i]
    hamming_df.at[i, 'phi'] = phi_test_NL[i]
    hamming_df.at[i, 'h'] = d
    #if d < min_d:
        #min_d = d
        #min_d_idx = i
#print('Closest:', min_d_idx)
wt_idx = 8214

hamming = hamming_df['h']

if 0:
    #plt.hist(hamming)
    plt.hist(hamming_df['h'])
    plt.show()

cmap = 'inferno'
norm = plt.Normalize(np.amin(np.array(hamming)), np.amax(np.array(hamming)))
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
wt_color = 'C3'#'cyan'
wt_size = 30

if 0:
    fig, axs = plt.subplots(1,2,figsize=[6,5])
    # linear:
    axs[0].scatter(phi_test_L, y_test_L,
            c=hamming, cmap='inferno', label='test set',
            s=1, alpha=.25, zorder=-100, rasterized=True)
    axs[0].scatter(phi_test_L[wt_idx], y_test_L[wt_idx], c=wt_color, s=wt_size, zorder=10, label='WT', edgecolors='k')
    axs[0].set_xlabel('Additive effect $\phi$\n(linear model)')
    axs[0].set_ylabel('DNN prediction $y$')
    x1,x2 = axs[0].get_xlim()
    y1,y2 = axs[0].get_ylim()
    # linear epistasis:
    phi_lim = [min(phi_test_L)-.5, max(phi_test_L)+.5] #set phi limit and create a grid in phi space
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
    yhat_grid = model_L.phi_to_yhat(phi_grid) #compute yhat each phi grid point
    q = [0.025, 0.975] #compute 95% CI for each yhat
    yqs_grid = model_L.yhat_to_yq(yhat_grid, q=q)
    axs[0].fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                   alpha=0.1, color='C1', lw=0, label=r'95$\%$ CI') #plot 95% confidence interval
    axs[0].plot(phi_grid, yhat_grid, linewidth=1.5, color='C1', label='GE') #plot GE nonlinearity
    axs[0].set_xlim(x1,x2)
    axs[0].set_ylim(y1,y2)
    # nonlinear:
    axs[1].scatter(phi_test_NL, y_test_NL,
            c=hamming, cmap=cmap, label='test set',
            s=1, alpha=.25, zorder=-100, rasterized=True)
    axs[1].scatter(phi_test_NL[wt_idx], y_test_NL[wt_idx], c=wt_color, s=wt_size, zorder=10, label='WT', edgecolors='k')
    axs[1].set_xlabel('Additive effect $\phi$\n(nonlinear model)')
    axs[1].set_ylabel('DNN prediction $y$')
    #nonlinear epistasis:
    phi_lim = [min(phi_test_NL)-.5, max(phi_test_NL)+.5] #set phi limit and create a grid in phi space
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
    yhat_grid = model_NL.phi_to_yhat(phi_grid) #compute yhat each phi grid point
    q = [0.025, 0.975] #compute 95% CI for each yhat
    yqs_grid = model_NL.yhat_to_yq(yhat_grid, q=q)
    axs[1].fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                   alpha=0.1, color='C1', lw=0, label=r'95$\%$ CI') #plot 95% confidence interval
    axs[1].plot(phi_grid, yhat_grid, linewidth=1.5, color='C1', label='GE') #plot GE nonlinearity
    #colorbar:
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'nonlinearity_view1.png'), facecolor='w', dpi=200)
    else:
        plt.show()

    
if 0:
    fig, axs = plt.subplots(1,3,figsize=[9,5])
    axs[0].scatter(phi_test_L, yhat_test_L,
            c=hamming, cmap='inferno',
            s=1, alpha=.25, label='test data',
            zorder=-100, rasterized=True)
    axs[0].scatter(phi_test_L[wt_idx], yhat_test_L[wt_idx], c=wt_color, s=wt_size, zorder=10, label='WT', edgecolors='k')
    axs[0].set_xlabel('Additive effect $\phi$\n(linear model)')
    axs[0].set_ylabel('Epistatic effect $\hat{y}$\n(linear model)')
    axs[1].scatter(phi_test_NL, yhat_test_NL,
            c=hamming, cmap='inferno',
            s=1, alpha=.25, label='test data',
            zorder=-100, rasterized=True)
    axs[1].scatter(phi_test_NL[wt_idx], yhat_test_NL[wt_idx], c=wt_color, s=wt_size, zorder=10, label='WT', edgecolors='k')
    axs[1].set_xlabel('Additive effect $\phi$\n(nonlinear model)')
    axs[1].set_ylabel('Epistatic effect $\hat{y}$\n(nonlinear model)')
    axs[2].scatter(phi_test_L, yhat_test_NL,
            c=hamming, cmap='inferno', 
            s=1, alpha=.25, label='test data',
            zorder=-100, rasterized=True)
    axs[2].scatter(phi_test_L[wt_idx], y_test_NL[wt_idx], c=wt_color, s=wt_size, zorder=10, label='WT', edgecolors='k')
    axs[2].set_xlabel('Additive effect $\phi$\n(linear model)')
    axs[2].set_ylabel('Epistatic effect $\hat{y}$\n(nonlinear model)')

    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    fig.tight_layout()
    plt.show()

if 1: #Fig. 2d
    color_add_L = '#f781bf' #pink
    color_add_NL = '#4daf4a' #green
    fig, axs = plt.subplots(1,2,figsize=[5,3])
    # linear:
    axs[0].scatter(yhat_test_L, y_test_L,
        c=color_add_L,
        s=1, alpha=.1, label='test set',
        zorder=-100, rasterized=True)
    #axs[0].scatter(yhat_test_L[wt_idx], y_test_L[wt_idx], c='k', s=7.5, zorder=100, label='WT')#, edgecolors='k')
    axs[0].set_xlabel('Epistatic effect $\hat{y}$\n(linear model)')
    axs[0].set_ylabel('DNN prediction $y$')
    # linear performance:
    #Rsq = np.corrcoef(yhat_test_L.ravel(), y_test_L['y'])[0, 1]**2 #compute R^2 between yhat_test and y_test    
    xlim = [min(yhat_test_L), max(yhat_test_L)]
    axs[0].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=1000)
    #axs[0].set_title(f'$R^2$={Rsq:.3}')
    #axs[0].legend(loc='upper left')
    # nonlinear:
    axs[1].scatter(yhat_test_NL, y_test_NL,
        #c=hamming, cmap='inferno',
        c=color_add_NL,
        s=1, alpha=.1, label='test data', #.25
        zorder=-100, rasterized=True)
    axs[1].scatter(yhat_test_NL[wt_idx], y_test_NL[wt_idx], c='k', s=7.5, zorder=100, label='WT')#, edgecolors='k')
    axs[1].set_xlabel('Epistatic effect $\hat{y}$\n(nonlinear model)')
    axs[1].set_yticks([])
    axs[1].yaxis.set_tick_params(labelleft=False)
    #axs[1].set_ylabel('DNN prediction $y$')
    # nonlinear performance:
    #Rsq = np.corrcoef(yhat_test_NL.ravel(), test_df_NL['y'])[0, 1]**2    
    xlim = [min(yhat_test_NL), max(yhat_test_NL)]
    axs[1].plot(xlim, xlim, '--', color='k', label='diagonal', zorder=1000)
    #axs[1].set_title(f'$R^2$={Rsq:.3}')

    axs[0].set_ylim(np.amin(y_test_L)-5, np.amax(y_test_L)+5)
    axs[1].set_ylim(np.amin(y_test_L)-5, np.amax(y_test_L)+5)

    #axs[1].legend(loc='upper left')
    # colorbar:
    #divider = make_axes_locatable(axs[1])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    fig.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'nonlinearity_view3.pdf'), facecolor='w', dpi=200) 
    plt.show()

if 0:
    fig, axs = plt.subplots(1,3,figsize=[9,3])
    sns.kdeplot(np.array(y_test_L), bw_adjust=0.01, ax=axs[0], color='C0')
    axs[0].set_xlabel('DNN prediction $y$\n(linear model)')
    xlim = axs[0].get_xlim()
    sns.kdeplot(np.array(phi_test_L), bw_adjust=0.01, ax=axs[1], color='C0')
    axs[1].set_xlabel('Additive effect $\phi$\n(linear model)')
    axs[1].set_xlim(xlim[0], xlim[1])
    sns.kdeplot(np.array(yhat_test_L), bw_adjust=0.01, ax=axs[2], color='C0')
    axs[2].set_xlabel('Epistatic effect $\hat{y}$\n(linear model)')
    axs[2].set_xlim(xlim[0], xlim[1])
    plt.tight_layout()
    plt.show()

if 0:
    fig, axs = plt.subplots(1,3,figsize=[9,3])
    sns.kdeplot(np.array(y_test_NL), bw_adjust=0.01, ax=axs[0], color='C1')
    axs[0].set_xlabel('DNN prediction $y$\n(nonlinear model)')
    xlim = axs[0].get_xlim()
    sns.kdeplot(np.array(phi_test_NL), bw_adjust=0.01, ax=axs[1], color='C1')
    axs[1].set_xlabel('Additive effect $\phi$\n(nonlinear model)')
    axs[1].set_xlim(xlim[0], xlim[1])
    sns.kdeplot(np.array(yhat_test_NL), bw_adjust=0.01, ax=axs[2], color='C1')
    axs[2].set_xlabel('Epistatic effect $\hat{y}$\n(nonlinear model)')
    axs[2].set_xlim(xlim[0], xlim[1])
    plt.tight_layout()
    plt.show()
    

if 1: #Fig. 2c
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    rocket0 = np.array([246/256, 180/256, 143/256, 1])
    rocket1 = np.array([243/256, 118/256, 81/256, 1])
    rocket2 = np.array([225/256, 51/256, 66/256, 1])
    rocket3 = np.array([173/256, 23/256, 89/256, 1])
    rocket4 = np.array([112/256, 31/256, 87/256, 1])
    rocket5 = np.array([53/256, 25/256, 62/256, 1])
    rocket6 = np.array([1/256, 1/256, 1/256, 1]) #black
    newcolors[:int(51.2), :] = rocket1
    newcolors[int(51.2):int(51.2*2), :] = rocket2
    newcolors[int(51.2*2):int(51.2*3), :] = rocket3
    newcolors[int(51.2*3):int(51.2*4), :] = rocket4
    newcolors[int(51.2*4):int(51.2*5), :] = rocket5
    newcolors[int(51.2*5):int(51.2*6), :] = rocket6

    cmap = ListedColormap(newcolors)
    norm = plt.Normalize(np.amin(np.array(hamming)), np.amax(np.array(hamming)))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axs = plt.subplots(1,1,figsize=[5,3]) #[4,5] #[8,3]

    #axs.scatter(phi_test_NL, y_test_NL,
            #c=hamming, cmap=cmap, 
            #s=1, alpha=.25, zorder=-100, rasterized=True)

    H0 = hamming_df.loc[hamming_df['h'] == 0]
    H1 = hamming_df.loc[hamming_df['h'] == 1]
    H2 = hamming_df.loc[hamming_df['h'] == 2]
    H3 = hamming_df.loc[hamming_df['h'] == 3]
    H4 = hamming_df.loc[hamming_df['h'] == 4]
    H5 = hamming_df.loc[hamming_df['h'] == 5]

    s = 15
    a = 1
    l = .3 #.1
    axs.scatter(H0['phi'], H0['y'],
            c=rocket1, linewidth=l, edgecolor=rocket0,
            s=s, alpha=a, zorder=-100, rasterized=True)
    axs.scatter(H1['phi'], H1['y'],
            c=rocket2, linewidth=l, edgecolor=rocket1,
            s=s, alpha=a, zorder=-90, rasterized=True)
    axs.scatter(H2['phi'], H2['y'],
            c=rocket3, linewidth=l, edgecolor=rocket2,
            s=s, alpha=a, zorder=-80, rasterized=True)
    axs.scatter(H3['phi'], H3['y'],
            c=rocket4, linewidth=l, edgecolor=rocket3,
            s=s, alpha=a, zorder=-70, rasterized=True)
    axs.scatter(H4['phi'], H4['y'],
            c=rocket5, linewidth=l, edgecolor=rocket4,
            s=s, alpha=a, zorder=-60, rasterized=True)
    axs.scatter(H5['phi'], H5['y'],
            c=rocket6, linewidth=l, edgecolor=rocket5,
            s=s, alpha=a, zorder=-50, rasterized=True)

    color_add = '#70cb6d'#'#4daf4a' #green
    axs.scatter(phi_test_NL[wt_idx], y_test_NL[wt_idx], c=wt_color, s=s, zorder=10, label='WT', edgecolors='k', linewidth=l)
    
    phi_lim = [min(phi_test_NL)-.5, max(phi_test_NL)+.5] #set phi limit and create a grid in phi space
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
    yhat_grid = model_NL.phi_to_yhat(phi_grid) #compute yhat each phi grid point
    q = [0.025, 0.975] #compute 95% CI for each yhat
    yqs_grid = model_NL.yhat_to_yq(yhat_grid, q=q)
    axs.fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                   alpha=.35, lw=0, label='95% CI', zorder=-300,
                   #color=rocket0,
                   color=color_add) #plot 95% confidence interval
    axs.plot(phi_grid, yhat_grid, linewidth=2, color=color_add, label='nonlinearity', zorder=-65) #plot GE nonlinearity

    axs.set_xlabel('Additive effect $\phi$')
    axs.set_ylabel('DNN prediction $y$')

    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical') 
    fig.tight_layout()

    if 1:
        plt.savefig(os.path.join(pyDir,'nonlinearity_v3.pdf'), facecolor='w', dpi=600)
    else:
        plt.show()

