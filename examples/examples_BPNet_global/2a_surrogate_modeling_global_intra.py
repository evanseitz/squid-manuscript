# =============================================================================
# Script for an example use of global surrogate modeling for a single motif..
# ..of interest (i.e., disregarding inter-motif relationships)
# (much less optimized than the main SQUID scripts in the parent directory)
# =============================================================================
# To use, first source the proper environment (i.e., 'conda activate mavenn')..
# ..select user settings below, and run via: python 2a_surrogate_modeling_global_intra.py
# =============================================================================

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mavenn
from IPython.display import display
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
motif_name = 'Oct4' #e.g., {'Oct4', 'Sox2', 'Klf4', 'Nanog'}
fname = 'MPRA_%s_N100000' % motif_name #must match filename of MAVE dataset output in '1a_generate_mave_global_intra.py'
gpmap = 'additive'


# load MPRA dataset
dataDir = os.path.join(pyDir, 'outputs/%s' % motif_name)
data_name = '%s.csv.gz' % fname
data_df = pd.read_csv(os.path.join(dataDir, data_name), compression='gzip')
data_df = data_df.dropna()
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
          verbose=False)

model.save(os.path.join(dataDir, '%s_GE_%s' % (fname[5:], gpmap)))

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
    fig.savefig(os.path.join(dataDir, '%s_training.png' % fname[5:]), facecolor="w", dpi=200)

if 1:
    theta_dict = model.get_theta(gauge='empirical') # fixes gauge; {'uniform', 'emperical', 'consensus', 'user'}
    theta_dict.keys() # returns: {'theta_0' constant term; 'theta_lc' LxC additive effects; 'theta_lclc' LxCxLxC pairwise effects; 'theta_bb' all params for blackbox}
    logo_df = theta_dict['logomaker_df']#*-1.
    logo_df.fillna(0, inplace=True)
    # create figure for logo
    fig, ax = plt.subplots(figsize=[10.5,1])
    logo = logomaker.Logo(df=logo_df,
                          ax=ax,
                          fade_below=.5,
                          shade_below=.5,
                          width=.9,
                          center_values=True,
                          font_name='Arial Rounded MT Bold')
    plt.xlim(475+5,525+5)
    plt.ylabel('MAVE-NN Pred')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ylim = ax.get_ylim()
    plt.show()
    fig.savefig(os.path.join(dataDir, '%s_logo.png' % fname[5:]), facecolor="w", dpi=600)