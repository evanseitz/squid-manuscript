# =============================================================================
# Script for an example use of global surrogate modeling for a pair of motifs..
# ..such that inter-motif relationships can be analyzed
# (much less optimized than the main SQUID scripts in the parent directory)
# =============================================================================
# To use, first source the proper environment (i.e., 'conda activate mavenn')..
# ..select user settings below, and run via (e.g.): 
#       python 2b_surrogate_modeling_global_inter.py 5
# ..where the positive integer 5 represents the desired inter-motif distance.
# Alternatively, a series of inter-motif instances can be run as a batch via:
#       bash 2b_batch.sh
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings


def op(pyDir, inter_dist): #'inter_dist' : distance between motifs (see below)

    import os, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mavenn
    import logomaker
    from matplotlib.ticker import MaxNLocator
    sys.dont_write_bytecode = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #turns off tensorflow warnings

    pyDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(pyDir)
    grandparentDir = os.path.dirname(parentDir)
    sys.path.append(pyDir)
    sys.path.append(parentDir)
    sys.path.append(grandparentDir)
    import squid.utils as squid_utils


    # user parameters:
    motifA_name = 'Nanog'
    motifB_name = 'Sox2'
    fname = 'MPRA_%s%s_N200000_dist%s' % (motifA_name, motifB_name, inter_dist) #must match filename of MAVE dataset output in '1b_generate_mave_global_inter.py'
    gpmap = 'pairwise'


    # load MPRA dataset
    dataDir = os.path.join(pyDir, 'outputs/%s%s' % (motifA_name, motifB_name))
    if not os.path.exists(os.path.join(dataDir, 'MAVENN')):
        os.mkdir(os.path.join(dataDir, 'MAVENN'))
    if not os.path.exists(os.path.join(dataDir, 'training')):
        os.mkdir(os.path.join(dataDir, 'training'))
    if not os.path.exists(os.path.join(dataDir, 'logos')):
        os.mkdir(os.path.join(dataDir, 'logos'))
    if not os.path.exists(os.path.join(dataDir, 'matrices')):
        os.mkdir(os.path.join(dataDir, 'matrices'))

    if 1:
        data_name = 'MPRA/%s.csv.gz' % fname
        data_df_pre = pd.read_csv(os.path.join(dataDir, data_name), compression='gzip')
        data_df_pre = data_df_pre.dropna()
    else: #if MPRA data saved among individual batches
        data_df_pre1 = pd.read_csv(os.path.join(dataDir, 'MPRA/%s_batch01.csv.gz' % fname),
                            compression='gzip')
        data_df_pre2 = pd.read_csv(os.path.join(dataDir, 'MPRA/%s_batch02.csv.gz' % fname),
                            compression='gzip')
        data_df_pre = pd.concat([data_df_pre1, data_df_pre2], axis=0, ignore_index=True)

    if 1: #method for delimiting sequence
        start, stop = 480, 535
        data_df = data_df_pre.copy()
        data_df['x'] = data_df['x'].str.slice(start,stop)
        #print(len(data_df['x'].iloc[0]))
        print(data_df.head())
    else:
        data_df = data_df_pre
        print(data_df.head())

    # split dataset
    trainval_df, test_df = mavenn.split_dataset(data_df)
    # show dataset sizes
    print(f'Train + val set size : {len(trainval_df):6,d} observations')
    print(f'Test set size        : {len(test_df):6,d} observations')
    # get the length of the sequence
    L = len(data_df['x'][0])
    print(f'Sequence length: {L:d} nucleotides')
    # Get the column index for the counts
    y_cols = trainval_df.columns[1:-1]
    # Get the number of count columns
    len_y_cols = len(y_cols)

    # =============================================================================
    # Training the Model
    # =============================================================================

    # define custom gp_map parameters dictionary
    gpmap_kwargs = {'L': L,
                    'C': 4,
                    'theta_regularization': 0.1}

    model = mavenn.Model(L=L,
                            Y=len_y_cols,
                            alphabet='dna',
                            regression_type='GE',
                            ge_noise_model_type='SkewedT',
                            ge_heteroskedasticity_order=2, 
                            gpmap_type=gpmap,
                            gpmap_kwargs=gpmap_kwargs);

    # set training data
    model.set_data(x=trainval_df['x'],
                    y=trainval_df['y'],
                    validation_flags=trainval_df['validation'],
                    shuffle=True);

    # fit model to data
    model.fit(learning_rate=5e-4, #5e-4 default
            epochs=500, #500 default
            batch_size=100,
            early_stopping=True,
            early_stopping_patience=25,
            linear_initialization=False,
            verbose=False);

    model.save(os.path.join(dataDir, 'MAVENN/%s_GE_%s' % (fname[5:], gpmap)))

    I_pred, dI_pred = model.I_predictive(x=test_df['x'], y=test_df['y'])
    print(f'test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits')

    if 1:
        fig, ax = plt.subplots(1, 1, figsize=[5, 5])
        ax.plot(model.history['I_var'], label=r'I_var_train')
        ax.plot(model.history['val_I_var'], label=r'val_I_var')
        ax.axhline(I_pred, color='C3', linestyle=':', label=r'test_I_pred')
        ax.set_xlabel('epochs')
        ax.set_ylabel('bits')
        ax.set_title('Training history: variational information')
        ax.legend()
        plt.tight_layout()
        #plt.show()
        fig.savefig(os.path.join(dataDir, 'training/training_%s_dist%s.png' % (fname[5:], inter_dist)), facecolor="w", dpi=200)
        plt.close()

    if 1:
        theta_dict = model.get_theta(gauge='empirical') # fixes gauge; {'uniform', 'emperical', 'consensus', 'user'}
        theta_dict.keys() # returns: {'theta_0' constant term; 'theta_lc' LxC additive effects; 'theta_lclc' LxCxLxC pairwise effects; 'theta_bb' all params for blackbox}
        logo_df = theta_dict['logomaker_df']*-1.
        logo_df.fillna(0, inplace=True)
        fig, ax = plt.subplots(figsize=[10,2])
        logo = logomaker.Logo(df=logo_df,
                            ax=ax,
                            fade_below=.5,
                            shade_below=.5,
                            width=.9,
                            center_values=True,
                            font_name='Arial Rounded MT Bold')
        plt.ylabel('MAVE-NN Pred')
        ylim = ax.get_ylim()
        #plt.show()
        fig.savefig(os.path.join(dataDir, 'logos/logo_%s_dist%s.png' % (fname[5:], inter_dist)), facecolor="w", dpi=600)
        plt.close()

    if gpmap == 'pairwise': #plot pairwise heatmap
        theta_lclc = theta_dict['theta_lclc']*-1.
        fig, ax = plt.subplots(figsize=[10,5])
        ax, cb = mavenn.heatmap_pairwise(values=theta_lclc,#[15:55,:,15:55,:],
                                        alphabet='dna',
                                        ax=ax,
                                        gpmap_type=gpmap,
                                        cmap_size='2%', #3%
                                        show_alphabet=False,
                                        cmap='seismic', #seismic, bwr
                                        cmap_pad=.1,
                                        show_seplines=True,            
                                        sepline_kwargs = {'color': 'k',
                                                        'linestyle': '-',
                                                        'linewidth': .5}) 
        ax.xaxis.set_ticks(np.arange(0, stop-start, 5))
        ax.set_xlabel(r'Nucleotide Position', labelpad=5)
        cb.set_label(r'Pairwise Effect',# ($\Delta \Delta \phi$)',
                    labelpad=8, ha='center', va='center', rotation=-90)
        cb.outline.set_visible(False)
        cb.ax.tick_params(direction='in', size=20, color='white')
        #plt.show()
        fig.savefig(os.path.join(dataDir, 'matrices/pairwise_%s_dist%s.png' % (fname[5:], inter_dist)), facecolor="w", dpi=600)
        plt.close()

if __name__ == '__main__':
    path1 = os.path.dirname(os.path.abspath(__file__))

    """
    df_idx : INT >= 0
        inter-motif distance between two motifs
    """
    
    if len(sys.argv) > 1:
        inter_dist = int(sys.argv[1])

    else:
        print('')
        print('Script must be run with trailing index argument:')
        print('e.g., 2b_surrogate_modeling_global_inter.py 5')
        print('')
        sys.exit(0)
    op(path1, inter_dist)