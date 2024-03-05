import os, sys
sys.dont_write_bytecode = True
import mavenn
import numpy as np

def run_ridge(mave, squid_utils, alphabet, drop=True):
    from sklearn.linear_model import RidgeCV

    if drop is True:
        mave.drop_duplicates(['y', 'x'], inplace=True)

    print('Encoding one hots...')
    X = np.zeros(shape=(mave['x'].shape[0], len(mave['x'][0]), len(alphabet)))
    for i in range(mave['x'].shape[0]):
        X[i,:,:] = squid_utils.seq2oh(mave['x'][i], alphabet)
    print('Running ridge regression...')
    
    Y = np.array(mave['y'])
    X = X.reshape(X.shape[0], -1)
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5).fit(X, Y)
    coef = model.coef_
    yhat = model.predict(X)

    if 1: #quantify ruggedness of function; i.e., how well a linear model can explain the fitness landscape
        s = np.mean(np.abs(coef))
        r = np.sqrt(np.mean((yhat - Y)**2))
        print('r/s: ', r/s) #roughness to slope ratio (r/s)

    return (coef, model, Y, yhat)


def run_lasso(mave, squid_utils, alphabet, drop=True, cv=False):
    from sklearn.linear_model import LassoCV, Lasso

    if drop is True:
        mave.drop_duplicates(['y', 'x'], inplace=True)

    print('Encoding one hots...')
    X = np.zeros(shape=(mave['x'].shape[0], len(mave['x'][0]), len(alphabet)))
    for i in range(mave['x'].shape[0]):
        X[i,:,:] = squid_utils.seq2oh(mave['x'][i], alphabet)
    print('Running lasso regression...')

    Y = np.array(mave['y'])
    X = X.reshape(X.shape[0], -1)
    if cv is False:
        model = Lasso(alpha=1.0).fit(X, Y)
    else:
        model = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5).fit(X, Y)
    coef = model.coef_
    yhat = model.predict(X)

    if 1: #quantify ruggedness of function; i.e., how well a linear model can explain the fitness landscape
        s = np.mean(np.abs(coef))
        r = np.sqrt(np.mean((yhat - Y)**2))
        print('r/s: ', r/s) #roughness to slope ratio (r/s)

    return (coef, model, Y, yhat)


def run_lime(mave, squid_utils, alphabet, k=10, drop=True, cv=False):
    from sklearn.linear_model import Lasso, LassoLarsCV
    # k specifies the number of nonzero weights

    if drop is True:
        mave.drop_duplicates(['y', 'x'], inplace=True)

    print('Encoding one hots...')
    X = np.zeros(shape=(mave['x'].shape[0], len(mave['x'][0]), len(alphabet)))
    for i in range(mave['x'].shape[0]):
        X[i,:,:] = squid_utils.seq2oh(mave['x'][i], alphabet)
    Y = np.array(mave['y'])
    X = X.reshape(X.shape[0], -1)

    print('Running LassoLars regression...')
    model = LassoLarsCV(cv=None).fit(X, Y)

    alphas = model.alphas_
    nonzero_weights = np.apply_along_axis(np.count_nonzero, 0, model.coef_path_)
    selected_alpha_idx = np.where(nonzero_weights == k)[0][0]
    selected_alpha = alphas[selected_alpha_idx]
    print('Selected alpha:', selected_alpha)

    # get the final coefficients
    coef = model.coef_path_[:,selected_alpha_idx]
    #yhat = np.dot(X, coef)

    # mask out all zeroed features
    zeros_index = np.where(coef==0)
    X_masked = X.copy()
    X_masked[:, zeros_index] = 0

    # refit the data on the masked set using least squares linear regression
    model_sparse = Lasso(alpha=0.).fit(X_masked, Y)
    coef_sparse = model_sparse.coef_
    yhat_sparse = model_sparse.predict(X)

    return (coef_sparse, model_sparse, Y, yhat_sparse)


def run_mavenn(mave, gpmap, alphabet, gauge, regression='GE', linearity='nonlinear',
                    noise='SkewedT', noise_order=2, drop=True):
    
    """
    MAVE-NN quantitative modeling of in-silico MAVE dataset
    
    mave:           dataframe of N partitioned ('set') sequences ('x')..
                    and their corresponding scores ('y');
                        shape (N, 3)
    gpmap:          genotype-phenotype map;
                        string: {'additive', 'neighbor', 'pairwise', 'blackbox', 'custom'}
    alphabet:       alphabet for sequence;
                        1D array: e.g., ['A', 'C', 'G', 'T']
    regression:     regression type for measurement process;
                        string: {'MPA', 'GE'}
                        - MPA:    measurement process agnostic (categorical y-values)
                        - GE:     global epistasis (continuous y-values)
    linearity:      linearity of measurement process;
                        string: {'linear', 'nonlinear'}
    noise:          GE noise models;
                        string: {'Gaussian', 'Cauchy', 'SkewedT'}
    noise_order:    in the GE model context, represents the order of the polynomial(s)..
                    used to define noise model parameters;
                        int >= 0
    drop:           drop all duplicate {x, y} rows in MAVE dataframe;
                        string: {True, False}
    """
    
    if drop is True:
        mave.drop_duplicates(['y', 'x'], inplace=True)
    
    # split dataset
    trainval_df, test_df = mavenn.split_dataset(mave)
    
    # show dataset sizes
    print(f'Train + val set size : {len(trainval_df):6,d} observations')
    print(f'Test set size        : {len(test_df):6,d} observations')
    
    # get the length of the sequence
    L = len(mave['x'][0])
    # get the column index for the counts
    y_cols = trainval_df.columns[1:-1]
    # get the number of count columns
    len_y_cols = len(y_cols)

    gpmap_kwargs = {'L': L,
                    'C': len(alphabet),
                    'theta_regularization': 0.1}
    
    if gpmap == 'additive':
        print('Initializing additive model... %s parameters' % int(L*4))
    elif gpmap == 'neighbor':
        print('Initializing neighbor model... %s parameters' % int((L-1)*16))
    elif gpmap == 'pairwise':
        print('Initializing pairwise model... %s parameters' % int((L*(L-1)*16)/2))
    
    # create the model
    if regression == 'MPA':
        model = mavenn.Model(L=L,
                             Y=len_y_cols,
                             alphabet='dna',
                             ge_nonlinearity_type=linearity,
                             regression_type='MPA',
                             gpmap_type=gpmap, 
                             gpmap_kwargs=gpmap_kwargs);
    
        # set training data
        model.set_data(x=trainval_df['x'],
                       y=trainval_df[y_cols],
                       validation_flags=trainval_df['validation'],
                       shuffle=True);
        
    elif regression == 'GE':
        model = mavenn.Model(L=L,
                             Y=len_y_cols,
                             alphabet='dna',
                             ge_nonlinearity_type=linearity,
                             regression_type='GE',
                             ge_noise_model_type=noise,
                             ge_heteroskedasticity_order=noise_order, 
                             gpmap_type=gpmap,
                             gpmap_kwargs=gpmap_kwargs);
    
        # set training data
        model.set_data(x=trainval_df['x'],
                       y=trainval_df['y'],
                       validation_flags=trainval_df['validation'],
                       shuffle=True);
    
    # fit model to data
    model.fit(learning_rate=5e-4,
              epochs=500,
              batch_size=100,
              early_stopping=True,
              early_stopping_patience=25,
              linear_initialization=False,
              #restore_best_weights=True,
              verbose=False);
    
    # Compute predictive information on test data
    if regression == 'MPA':
        I_pred, dI_pred = model.I_predictive(x=test_df['x'], y=test_df[y_cols])
    elif regression == 'GE':
        I_pred, dI_pred = model.I_predictive(x=test_df['x'], y=test_df['y'])
    print(f'test_I_pred: {I_pred:.3f} +- {dI_pred:.3f} bits')
    print('max I_var:',np.amax(model.history['I_var']))
    print('max val_I_var:',np.amax(model.history['val_I_var']))

    if 1: #quantify ruggedness of function; i.e., how well a linear model can explain the fitness landscape
        theta_dict = model.get_theta(gauge=gauge)
        theta_lc = theta_dict['theta_lc']
        s = np.mean(np.abs(theta_lc))
        y = trainval_df['y'] #get test data y values
        yhat = model.x_to_yhat(trainval_df['x']) #compute yhat on test data
        r = np.sqrt(np.mean((yhat - y)**2))
        print('r/s: ', r/s) #roughness to slope ratio (r/s)


    return (model, I_pred)