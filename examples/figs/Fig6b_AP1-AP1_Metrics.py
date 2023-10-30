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
from scipy import stats
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

alphabet = ['A','C','G','T']

'''custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
color_palette = ['#f781bf', '#4daf4a']
sns.set_palette(palette=color_palette, n_colors=2)'''


motif_name = '13_AP1_N_13_AP1_N'
model_name1 = 'GOPHER'
model_name2 = 'ResidualBind32_Exp_single'
folder_NL = 'SQUID_13_AP1_N_13_AP1_N_inter_mut0'
nS = 49


if 0: #calculate yhat from files (run if first time using script)
    load = False
    df = pd.DataFrame(columns = ['Model', 'Metric', 'Value'], index=range(nS))
else: #load in pre-calculated yhats
    load = True
    df = pd.read_csv(os.path.join(pyDir, 'Fig6_data/AP1-AP1_metrics.csv'))
    metric = 'PCC'##'PCC' #{MAE, MSE, RMSE, r, R2, PCC, r/s}


if load is False:
    idx = 0
    for df_idx in range(nS+1):
        print(df_idx)

        # load in models/parameters:
        dir = os.path.join(parentDir, 'examples_%s/c_surrogate_outputs/model_%s/%s' % (model_name1, model_name2, folder_NL))
        for folder in os.listdir(dir):
            if folder.startswith('rank%s_' % (df_idx)):
                path = os.path.join(dir, folder)

        model_add0 = mavenn.load(os.path.join(path, '1_ADD_L/mavenn_model_linear'))
        model_add1 = mavenn.load(os.path.join(path, '2_ADD_GE/mavenn_model'))
        model_pw0 = mavenn.load(os.path.join(path, '3_PW_L/mavenn_model_linear'))
        model_pw1 = mavenn.load(os.path.join(path, '4_PW_GE/mavenn_model'))

        # load in MAVE dataset:
        mave = pd.read_pickle(os.path.join(path, 'mave_preds_unwrapped.csv.gz'), compression='gzip')
        trainval_df, test_df = mavenn.split_dataset(mave)
        X = mave['x']
        Y = np.array(mave['y'])

        # calculate yhats:
        X_OH = np.zeros(shape=(mave['x'].shape[0], len(mave['x'][0]), len(alphabet)))
        for i in range(mave['x'].shape[0]):
            X_OH[i,:,:] = squid_utils.seq2oh(mave['x'][i], alphabet)
        X_OH = X_OH.reshape(X_OH.shape[0], -1)

        Yhat_add0 = model_add0.x_to_yhat(X)
        Yhat_add1 = model_add1.x_to_yhat(X)
        Yhat_pw0 = model_pw0.x_to_yhat(X)
        Yhat_pw1 = model_pw1.x_to_yhat(X)

        # mean absolute error:
        MAE_add0 = (1./nS)*sum(np.abs(Yhat_add0 - Y)) #MAE: perfect=0; outliers not penalized
        MAE_add1 = (1./nS)*sum(np.abs(Yhat_add1 - Y)) 
        MAE_pw0 =(1./nS)*sum(np.abs(Yhat_pw0 - Y))
        MAE_pw1 =(1./nS)*sum(np.abs(Yhat_pw1 - Y))

        df.at[idx, 'Model'] = 'ADD_L'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_add0
        idx += 1
        df.at[idx, 'Model'] = 'ADD_GE'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_add1
        idx += 1
        df.at[idx, 'Model'] = 'PW_L'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_pw0
        idx += 1
        df.at[idx, 'Model'] = 'PW_GE'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_pw1
        idx += 1
        
        # mean square error:
        MSE_add0 = (1./nS)*sum((Yhat_add0 - Y)**2) #MSE: perfect=0; outlier penalized exponentially
        MSE_add1  =(1./nS)*sum((Yhat_add1 - Y)**2)
        MSE_pw0 = (1./nS)*sum((Yhat_pw0 - Y)**2)
        MSE_pw1 = (1./nS)*sum((Yhat_pw1 - Y)**2)
        df.at[idx, 'Model'] = 'ADD_L'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_add0
        idx += 1
        df.at[idx, 'Model'] = 'ADD_GE'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_add1
        idx += 1
        df.at[idx, 'Model'] = 'PW_L'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_pw0
        idx += 1
        df.at[idx, 'Model'] = 'PW_GE'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_pw1
        idx += 1

        # root mean square error:
        RMSE_add0 = np.sqrt((1./nS)*sum((Yhat_add0 - Y)**2)) #RMSE: use if data has asymetric conditional distribution
        RMSE_add1 = np.sqrt((1./nS)*sum((Yhat_add1 - Y)**2))
        RMSE_pw0 = np.sqrt((1./nS)*sum((Yhat_pw0 - Y)**2))
        RMSE_pw1 = np.sqrt((1./nS)*sum((Yhat_pw1 - Y)**2))
        df.at[idx, 'Model'] = 'ADD_L'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_add0
        idx += 1
        df.at[idx, 'Model'] = 'ADD_GE'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_add1
        idx += 1
        df.at[idx, 'Model'] = 'PW_L'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_pw0
        idx += 1
        df.at[idx, 'Model'] = 'PW_GE'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_pw1
        idx += 1

        # Pearson correlation:
        r_add0 = stats.pearsonr(Yhat_add0, Y)[0]
        r_add1 = stats.pearsonr(Yhat_add1, Y)[0]
        r_pw0 = stats.pearsonr(Yhat_pw0, Y)[0]
        r_pw1 = stats.pearsonr(Yhat_pw1, Y)[0]
        df.at[idx, 'Model'] = 'ADD_L'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_add0
        idx += 1
        df.at[idx, 'Model'] = 'ADD_GE'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_add1
        idx += 1
        df.at[idx, 'Model'] = 'PW_L'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_pw0
        idx += 1
        df.at[idx, 'Model'] = 'PW_GE'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_pw1
        idx += 1

        # Pearson product-moment correlation coefficients:
        PCC_add0 = np.corrcoef(Yhat_add0, Y)[0, 1]**2
        PCC_add1 = np.corrcoef(Yhat_add1, Y)[0, 1]**2
        PCC_pw0 = np.corrcoef(Yhat_pw0, Y)[0, 1]**2
        PCC_pw1 = np.corrcoef(Yhat_pw1, Y)[0, 1]**2
        df.at[idx, 'Model'] = 'ADD_L'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_add0
        idx += 1
        df.at[idx, 'Model'] = 'ADD_GE'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_add1
        idx += 1
        df.at[idx, 'Model'] = 'PW_L'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_pw0
        idx += 1
        df.at[idx, 'Model'] = 'PW_GE'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_pw1
        idx += 1

    df.to_csv(os.path.join(pyDir, 'Fig6_data/AP1-AP1_metrics.csv'), index=False)


else:
    df = df.loc[(df['Metric'] == metric)]
    nS = df.shape[0]

    if 1:
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        color_add_L = '#f781bf' #pink
        color_add_GE = '#4daf4a' #green
        color_palette = [color_add_L, color_add_GE]
        sns.set_palette(palette=color_palette, n_colors=2)
    else:
        colors = np.arange(nS)


    fig, ax = plt.subplots(figsize=(5, 4))
    palette = sns.color_palette("rocket", as_cmap=True, n_colors=nS)
    if 0:
        box = sns.boxplot(ax=ax, x=df['Model'], y=df['Value'], dodge=True, width=0.55, showmeans=True,
                        meanprops={"markerfacecolor":"black", "markeredgecolor":"black", "markersize":"10"})
    elif 0:
        violin = sns.violinplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32), hue=df['Model'], cut=0)
    elif 1:
        sns.swarmplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32), hue=df['Model'])#, palette=palette), hue=colors)
    else:
        sns.stripplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32))#, palette=palette, hue=colors, s=2)#, alpha=.25)

    if metric == 'r' or metric == 'PCC':
        ax.set_ylim(0,1)
    if metric == 'R2':
        ax.set_ylim(-1,1)
    else:
        y1, y2 = ax.get_ylim()
        ax.set_ylim(0,y2)

    if 1: #calculate P value
        add_1 = df.loc[df['Model'] == 'ADD_GE']
        pw_1 = df.loc[df['Model'] == 'PW_GE']
        from scipy import stats
        mwu_stat, pval = stats.mannwhitneyu(list(add_1['Value']), list(pw_1['Value']), alternative='two-sided')
        print('p value:', pval)


    plt.legend([],[], frameon=False)
    plt.title(metric)
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'Fig6_data/regression_metrics.pdf'), facecolor='w', dpi=200)
    plt.show()