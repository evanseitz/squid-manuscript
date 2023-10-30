import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import pandas as pd
import mavenn
import seaborn as sns
import joblib

#use environment: 'mavenn_citra'

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

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
#color_palette = ['#f781bf', '#4daf4a']
#sns.set_palette(palette=color_palette, n_colors=2)

color_add_L = '#f781bf' #pink
color_add_NL = '#4daf4a' #green

alphabet = ['A','C','G','T']

fig, axs = plt.subplots(2,2,figsize=[5,5])
dirs = ['examples_CAGI5/c_surrogate_outputs/model_Basenji32_GELU_single/SQUID_CAGI5-GOPHER_intra_mut0/rank11_seq11',
        'examples_CAGI5/c_surrogate_outputs/model_Basenji32_GELU_single/SQUID_CAGI5-GOPHER_intra_mut0/rank11_seq11',
        'examples_CAGI5/c_surrogate_outputs/model_Basenji32_GELU_single/SQUID_CAGI5-GOPHER_intra_mut0/rank12_seq12',
        'examples_CAGI5/c_surrogate_outputs/model_Basenji32_GELU_single/SQUID_CAGI5-GOPHER_intra_mut0/rank12_seq12']
models = ['ridge','GE','ridge','GE']
cols = [color_add_L, color_add_NL, color_add_L, color_add_NL]
ax_X = [0,0,1,1]
ax_Y = [0,1,0,1]
for idx in range(len(dirs)):
    dir = os.path.join(parentDir, dirs[idx])
    if models[idx] == 'ridge':
        model = joblib.load(os.path.join(dir,'ridge_model.pkl'))
        mave = pd.read_pickle(os.path.join(dir, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
        X = mave['x']
        Y = np.array(mave['y'])
        # calculate yhats:
        X_OH = np.zeros(shape=(mave['x'].shape[0], len(mave['x'][0]), len(alphabet)))
        for i in range(mave['x'].shape[0]):
            X_OH[i,:,:] = squid_utils.seq2oh(mave['x'][i], alphabet)
        X_OH = X_OH.reshape(X_OH.shape[0], -1)
        Yhat = model.predict(X_OH)
        xlim = [min(Yhat), max(Yhat)]
        axs[ax_X[idx], ax_Y[idx]].plot(xlim, xlim,
                linewidth=1, color='k', label='linearity', zorder=10) #plot GE nonlinearity
        axs[ax_X[idx], ax_Y[idx]].scatter(Yhat, Y,
                    color=cols[idx], s=1, alpha=.1, label='test data',
                    zorder=0, rasterized=True)

    elif models[idx] == 'GE':
        model = mavenn.load(os.path.join(dir, 'mavenn_model'))
        mave = pd.read_pickle(os.path.join(dir, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
        trainval_df, test_df = mavenn.split_dataset(mave)
        phi_test = model.x_to_phi(test_df['x']) #compute φ on test data
        phi_lim = [min(phi_test)-.5, max(phi_test)+.5] #set phi limit and create a grid in phi space
        phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
        yhat_grid = model.phi_to_yhat(phi_grid) #compute yhat each phi grid point
        #q = [0.025, 0.975] #compute 95% CI for each yhat
        #yqs_grid = model.yhat_to_yq(yhat_grid, q=q)
        #axs[ax_X[idx], ax_Y[idx]].fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                        #alpha=0.1, color='#4b4b4b', lw=0, label='95% CI', zorder=-100) #plot 95% confidence interval
        axs[ax_X[idx], ax_Y[idx]].plot(phi_grid, yhat_grid,
                linewidth=1, color='k', label='nonlinearity', zorder=10) #plot GE nonlinearity
        y_test = test_df['y']
        axs[ax_X[idx], ax_Y[idx]].scatter(phi_test, y_test,
                    color=cols[idx], s=1, alpha=.1, label='test data',
                    zorder=0, rasterized=True) #plot scatter of φ and y values


#plt.xlim(-5,2)
fig.tight_layout()
if 1:
    plt.savefig(os.path.join(pyDir, 'linear_vs_nonlinear.pdf'), facecolor='w', dpi=200)
plt.show()


