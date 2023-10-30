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


if 0:
    motif_name = 'DRE-long'
    model_name1 = 'DeepSTARR'
    model_name2 = 'DeepSTARR'
    folder_NL = 'SQUID_DRE-long_intra_mut0_M337_N25k'
    nS = 337
    phi = False
elif 0:
    motif_name = 'LOW'
    model_name1 = 'DeepSTARR'
    model_name2 = 'DeepSTARR'
    folder_NL = 'SQUID_LOW_intra_mut0_M300_N25k'
    nS = 299
    phi = False
elif 1:
    motif_name = '13_AP1'
    model_name1 = 'GOPHER'
    model_name2 = 'ResidualBind32_ReLU_single'
    if 0:
        #folder_NL = 'SQUID_13_AP1_intra_mut0_M300_N25k_PCA'
        folder_NL = 'SQUID_13_AP1_intra_mut0_M300_N100k_sum'
        nS = 300
        phi = True
    else:
        folder_NL = 'SQUID_13_AP1_intra_mut0'
        nS = 49
        phi = False


if 0: #calculate yhat from files (run if first time using script)
    load = False
    df = pd.DataFrame(columns = ['Model', 'Metric', 'Value'], index=range(nS))
else: #load in pre-calculated yhats
    load = True
    if 1:
        df = pd.read_csv(os.path.join(pyDir, 'Fig2_data/%s_metrics.csv' % (motif_name)))
        #df = pd.read_csv(os.path.join(pyDir, 'Fig2_data/%s_metrics_PCA.csv' % (motif_name)))
        joined = False
    else: #combine two pre-saved dataframes
        df1 = pd.read_csv(os.path.join(pyDir, 'Fig2_data/DRE-long_metrics.csv')) #len 337
        df2 = pd.read_csv(os.path.join(pyDir, 'Fig2_data/LOW_metrics.csv')) #len 299
        df1['Color'] = 'DRE'
        df2['Color'] = 'low y'
        df = pd.concat([df1, df2])
        joined = True
    metric = 'PCC' #{MAE, MSE, RMSE, r, R2, PCC, r/s}



if load is False:
    idx = 0
    for df_idx in range(nS+1):
        print(df_idx)

        # load in models/parameters:
        dir = os.path.join(parentDir, 'examples_%s/c_surrogate_outputs/model_%s/%s' % (model_name1, model_name2, folder_NL))
        for folder in os.listdir(dir):
            if folder.startswith('rank%s_' % (df_idx)):
                path = os.path.join(dir, folder)
        model_R = joblib.load(os.path.join(path,'ridge_model.pkl'))
        if phi is True:
            model_L = mavenn.load(os.path.join(path, 'mavenn_model_linear'))
        model_NL = mavenn.load(os.path.join(path, 'mavenn_model'))

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
        Yhat_ridge = model_R.predict(X_OH)

        if phi is True:
            Yhat_phi = model_L.x_to_yhat(X)
        else:
            Yhat_phi = model_NL.x_to_phi(X)
        Yhat_GE = model_NL.x_to_yhat(X)

        if 0:
            xlim = [min(Yhat_ridge), max(Yhat_ridge)]
            plt.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
            plt.scatter(Yhat_ridge, Y, alpha=.05, s=1)
            plt.show()
            xlim = [min(Yhat_phi), max(Yhat_phi)]
            plt.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
            plt.scatter(Yhat_phi, Y, alpha=.05, s=1)
            plt.show()
            xlim = [min(Yhat_GE), max(Yhat_GE)]
            plt.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
            plt.scatter(Yhat_GE, Y, alpha=.05, s=1)
            plt.show()

        # mean absolute error:
        MAE_ridge = (1./nS)*sum(np.abs(Yhat_ridge - Y)) #MAE: perfect=0; outliers not penalized
        MAE_phi = (1./nS)*sum(np.abs(Yhat_phi - Y)) 
        MAE_GE =(1./nS)*sum(np.abs(Yhat_GE - Y))
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_ridge
        idx += 1
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_phi
        idx += 1
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'MAE'
        df.at[idx, 'Value'] = MAE_GE
        idx += 1
        
        # mean square error:
        MSE_ridge = (1./nS)*sum((Yhat_ridge - Y)**2) #MSE: perfect=0; outlier penalized exponentially
        MSE_phi  =(1./nS)*sum((Yhat_phi - Y)**2)
        MSE_GE = (1./nS)*sum((Yhat_GE - Y)**2)
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_ridge
        idx += 1
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_phi
        idx += 1
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'MSE'
        df.at[idx, 'Value'] = MSE_GE
        idx += 1

        # root mean square error:
        RMSE_ridge = np.sqrt((1./nS)*sum((Yhat_ridge - Y)**2)) #RMSE: use if data has asymetric conditional distribution
        RMSE_phi = np.sqrt((1./nS)*sum((Yhat_phi - Y)**2))
        RMSE_GE = np.sqrt((1./nS)*sum((Yhat_GE - Y)**2))
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_ridge
        idx += 1
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_phi
        idx += 1
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'RMSE'
        df.at[idx, 'Value'] = RMSE_GE
        idx += 1

        # Pearson correlation:
        r_ridge = stats.pearsonr(Yhat_ridge, Y)[0]
        r_phi = stats.pearsonr(Yhat_phi, Y)[0]
        r_GE = stats.pearsonr(Yhat_GE, Y)[0]
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_ridge
        idx += 1
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_phi
        idx += 1
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'r'
        df.at[idx, 'Value'] = r_GE
        idx += 1

        # coefficient of determination:
        RSS = ((Y - Yhat_ridge)** 2).sum()
        TSS = ((Y - Y.mean()) ** 2).sum()
        R2_ridge = 1 - (RSS/TSS)
        #print(R2_ridge, model_R.score(X_OH, Y))
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'R2'
        df.at[idx, 'Value'] = R2_ridge
        idx += 1
        RSS = ((Y - Yhat_phi)** 2).sum()
        R2_phi = 1 - (RSS/TSS)
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'R2'
        df.at[idx, 'Value'] = R2_phi
        idx += 1
        RSS = ((Y - Yhat_GE)** 2).sum()
        R2_GE = 1 - (RSS/TSS)
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'R2'
        df.at[idx, 'Value'] = R2_GE
        idx += 1

        # Pearson product-moment correlation coefficients:
        PCC_ridge = np.corrcoef(Yhat_ridge, Y)[0, 1]**2
        PCC_phi = np.corrcoef(Yhat_phi, Y)[0, 1]**2
        PCC_GE = np.corrcoef(Yhat_GE, Y)[0, 1]**2
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_ridge
        idx += 1
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_phi
        idx += 1
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'PCC'
        df.at[idx, 'Value'] = PCC_GE
        idx += 1

        # roughness to slope ratio (r/s)
        coef = model_R.coef_
        s = np.mean(np.abs(coef))
        r = np.sqrt(np.mean((Yhat_ridge - Y)**2))
        df.at[idx, 'Model'] = 'Ridge'
        df.at[idx, 'Metric'] = 'r/s'
        df.at[idx, 'Value'] = r/s
        idx += 1
        if phi is True:
            theta_dict_L = model_L.get_theta(gauge='empirical')
            s = np.mean(np.abs(theta_dict_L['theta_lc']))
            r = np.sqrt(np.mean((Yhat_phi - Y)**2))
        df.at[idx, 'Model'] = 'phi'
        df.at[idx, 'Metric'] = 'r/s'
        df.at[idx, 'Value'] = r/s
        idx += 1
        theta_dict_NL = model_NL.get_theta(gauge='empirical')
        s = np.mean(np.abs(theta_dict_NL['theta_lc']))
        r = np.sqrt(np.mean((Yhat_GE - Y)**2))
        df.at[idx, 'Model'] = 'GE'
        df.at[idx, 'Metric'] = 'r/s'
        df.at[idx, 'Value'] = r/s
        idx += 1


    df.to_csv(os.path.join(pyDir, 'Fig2_data/%s_metrics.csv' % (motif_name)), index=False)


else:
    if phi is False:
        df.drop(df[df['Model'] == 'phi'].index, inplace=True)
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


    fig, ax = plt.subplots(figsize=(3, 3))
    palette = sns.color_palette("rocket", as_cmap=True, n_colors=nS)
    if 0:
        box = sns.boxplot(ax=ax, x=df['Model'], y=df['Value'], dodge=True, width=0.55, showmeans=True,
                        meanprops={"markerfacecolor":"black", "markeredgecolor":"black", "markersize":"10"})
    elif 0:
        violin = sns.violinplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32), hue=df['Model'], cut=0)
    elif 1:
        if joined is False:
            sns.swarmplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32), hue=df['Model'])#, palette=palette), hue=colors)
        else:
            sns.swarmplot(ax=ax, x=df['Model'], y=df['Value'].astype(np.float32), hue=df['Color'])
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
        add_GE = df.loc[df['Model'] == 'GE']
        add_R = df.loc[df['Model'] == 'Ridge']
        from scipy import stats
        print(add_GE['Value'])
        print(add_R['Value'])
        mwu_stat, pval = stats.mannwhitneyu(list(add_GE['Value']), list(add_R['Value']), alternative='two-sided')
        print('P value (NL–R):', pval)


    if joined is False:
        plt.legend([],[], frameon=False)
    plt.title(metric)
    plt.tight_layout()
    if 1:
        plt.savefig(os.path.join(pyDir,'Fig2_data/regression_metrics.pdf'), facecolor='w', dpi=200)
    plt.show()