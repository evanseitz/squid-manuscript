import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import squid.utils as squid_utils
import logomaker
import mavenn


# save record of user parameters required for '3_surrogate_modelin.py'
def params_info(surrogate, regression, gpmap,
                gauge, linearity, noise, noise_order, drop,
                saveDir):
    if os.path.exists(os.path.join(saveDir,'parameters_2.txt')):
        os.remove(os.path.join(saveDir,'parameters_2.txt'))
    f_out = open(os.path.join(saveDir,'parameters_2.txt'),'w')
    print('surrogate: %s' % (surrogate), file=f_out)
    print('regression: %s' % (regression), file=f_out)
    print('gpmap: %s' % (gpmap), file=f_out)
    print('gauge: %s' % (gauge), file=f_out)
    print('linearity: %s' % (linearity), file=f_out)
    print('noise: %s' % (noise), file=f_out)
    print('noise_order: %s' % (noise_order), file=f_out)
    print('drop: %s' % (drop), file=f_out)
    print('', file=f_out)
    f_out.close()


# plot single attribution map
def single_logo(logo, logo_name, ccenter, start, stop, figpad, saveDir):         
    fig, ax = plt.subplots()
    logomaker.Logo(df=logo[start-figpad:stop+figpad],
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=ccenter,
                    color_scheme='classic',
                    font_name='Arial Rounded MT Bold')
    ax.set_title('%s' % logo_name, fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'attr_logo_%s_figpad%s.png' % (logo_name, figpad)), facecolor='w', dpi=200)
    plt.close()
    

# plot double ISM matrix
def ISM_double(ISM_matrix, start, stop, pad, alphabet, saveDir):
    fig, ax = plt.subplots(figsize=[10,5])
    ax, cb = mavenn.heatmap_pairwise(values=ISM_matrix,
                                    alphabet=alphabet,
                                    ax=ax,
                                    ccenter=True,
                                    gpmap_type='pairwise',
                                    cmap_size='2%', #3%
                                    show_alphabet=False,
                                    cmap='inferno',
                                    cmap_pad=.1,
                                    #clim=[-10,10],
                                    show_seplines=True,            
                                    sepline_kwargs = {'color': 'white',
                                                      'linestyle': '-',
                                                      'linewidth': 2}) 
    skip = 2 #skip every nth xtick label (for visualization only)
    if pad != 'full':
        ax.xaxis.set_ticks(np.arange(0,stop+pad-(start-pad),skip))
        ax.set_xticklabels(np.arange(start-pad,stop+pad,skip))
    cb.set_label(r'Double ISM',
                labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'attributions_ISM_double.png'), facecolor='w', dpi=200)
    plt.close()
    

# save mavenn model information metrics to text
def mavenn_info(mavenn_model,I_pred, saveDir):
    if os.path.exists(os.path.join(saveDir,'mavenn_info.txt')):
        os.remove(os.path.join(saveDir,'mavenn_info.txt'))
    f_out = open(os.path.join(saveDir,'mavenn_info.txt'),'w')
    print(f'test_I_pred: {I_pred:.3f} bits', file=f_out)
    print('max I_var:',np.amax(mavenn_model.history['I_var']), file=f_out)
    print('max val_I_var:',np.amax(mavenn_model.history['val_I_var']), file=f_out)
    print('', file=f_out)
    f_out.close()
    

# plot mavenn model performance
def mavenn_performance(mavenn_model, I_pred, saveDir):
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    # plot I_var_train, the variational information on training data as a function of epoch
    ax.plot(mavenn_model.history['I_var'], label=r'I_var_train')
    # plot I_var_val, the variational information on validation data as a function of epoch
    ax.plot(mavenn_model.history['val_I_var'], label=r'val_I_var')
    # plot I_pred_test, the predictive information of the final model on test data
    ax.axhline(I_pred, color='C3', linestyle=':', label=r'test_I_pred')
    ax.set_xlabel('epochs')
    ax.set_ylabel('bits')
    ax.set_title('Training history')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, 'mavenn_training.png'), facecolor='w', dpi=200)
    plt.close()
    

# plot close-up of additive logo at chosen motif instance
def logo_additive(logo, scope, start, stop, figpad, saveDir, saveName):
    fig, ax = plt.subplots(figsize=[10,3]) #[5,1.5]) 
    logo_fig = logo[start-figpad:stop+figpad]
    logomaker.Logo(df=logo_fig,
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=True,
                    font_name='Arial Rounded MT Bold')
    ax.set_ylabel('Additive effect')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'%s_additive_figpad%s.png' % (saveName, figpad)), facecolor='w', dpi=200)
    plt.close()
    
    
# plot wideshot comparison of logos 
def compare_logos_ws(ISM_logo, sal_logo, mavenn_logo, scope, start, stop, pad, halfwidth, num_sim, maxL, map_crop, saveDir):
    logo_count = 0
    if ISM_logo is not None:
        logo_count += 1
    if sal_logo is not None:
        logo_count += 1
        
    if start - halfwidth < 0:
        halfwidth = start
    if stop + halfwidth > maxL:
        halfwidth = maxL-stop

    fig, axs = plt.subplots(nrows=logo_count+1, ncols=1, figsize=[100,20])

    axIdx = 0
    if ISM_logo is not None: #gold-standard ISM
        logomaker.Logo(df=ISM_logo[start-halfwidth:stop+halfwidth],
                              ax=axs[axIdx],
                              fade_below=.5,
                              shade_below=.5,
                              width=.9,
                              center_values=True,
                              font_name='Arial Rounded MT Bold')
        axs[axIdx].set_title('ISM', fontsize=48)
        axs[axIdx].axvspan(start-0.5, stop-0.5, alpha=.1, color='magenta', zorder=-10)
        axs[axIdx].get_xaxis().set_ticks([])
        axs[axIdx].tick_params(axis='y', which='major', labelsize=30)
        if map_crop is True and pad != 'full':
            axs[axIdx].axvspan(0-0.5, (start-pad)-0.5, alpha=.1, color='gray', zorder=-10)
            axs[axIdx].axvspan((stop+pad)-0.5, maxL-0.5, alpha=.1, color='gray', zorder=-10)
        axIdx += 1
    
    if sal_logo is not None: #saliency map
        logomaker.Logo(df=sal_logo[start-halfwidth:stop+halfwidth],
                              ax=axs[axIdx],
                              fade_below=.5,
                              shade_below=.5,
                              width=.9,
                              center_values=True,
                              font_name='Arial Rounded MT Bold')
        axs[axIdx].set_title('Saliency', fontsize=48)
        axs[axIdx].axvspan(start-0.5, stop-0.5, alpha=.1, color='magenta', zorder=-10)
        axs[axIdx].tick_params(axis='y', which='major', labelsize=30)
        axIdx += 1

    # mavenn additive model:
    axs[axIdx-1].get_xaxis().set_ticks([])
    logomaker.Logo(df=mavenn_logo[start-halfwidth:stop+halfwidth],
                          ax=axs[axIdx],
                          fade_below=.5,
                          shade_below=.5,
                          width=.9,
                          center_values=True,
                          font_name='Arial Rounded MT Bold')
    axs[axIdx].set_title(r'MAVE-NN Additive | $N$ = %s' % num_sim, fontsize=48)
    axs[axIdx].axvspan(start-0.5, stop-0.5, alpha=.1, color='magenta', zorder=-10)
    axs[axIdx].tick_params(axis='x', which='major', labelsize=60)
    axs[axIdx].tick_params(axis='y', which='major', labelsize=30)
    if pad != 'full':
        axs[axIdx].axvspan(0-0.5, (start-pad)-0.5, alpha=.1, color='gray', zorder=-10)
        axs[axIdx].axvspan((stop+pad)-0.5, maxL-0.5, alpha=.1, color='gray', zorder=-10)

    plt.tight_layout()
    plt.savefig(os.path.join(saveDir, 'attributions_width%s.png' % (int(halfwidth*2))), facecolor='w', dpi=200)
    plt.close()
    
    
# plot maveen pairwise matrix
def mavenn_pairwise(theta_lclc, start, stop, pad, alphabet, saveDir):
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
                                                      'linewidth': .5})
    if pad != 'full':
        ax.xaxis.set_ticks(np.arange(0,(stop+pad)-(start-pad),2))
        ax.set_xticklabels(np.arange((start-pad),(stop+pad),2))  
    cb.set_label(r'Pairwise Effect',
                  labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')
    plt.tight_layout()
    if saveDir is not None:
        plt.savefig(os.path.join(saveDir,'mavenn_pairwise.png'), facecolor='w', dpi=200)
        plt.close()
    

# plot y versus yhat
def y_vs_yhat(model, y, yhat, saveDir):
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    Rsq = np.corrcoef(yhat, y)[0, 1]**2 #compute R^2 between yhat and y
    ax.scatter(yhat, y, color='C0', s=1, alpha=.1,
               label='test data')
    xlim = [min(yhat), max(yhat)]
    ax.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
    ax.set_xlabel('model prediction ($\hat{y}$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title(f'Standard metric of model performance:\n$R^2$={Rsq:.3}')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'ridge_measure_yhat.png'), facecolor='w', dpi=200)
    plt.close()


# plot mavenn y versus yhat
def mavenn_yhat(mavenn_model, test_df, saveDir):
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    y_test = test_df['y'] #get test data y values
    yhat_test = mavenn_model.x_to_yhat(test_df['x']) #compute yhat on test data
    Rsq = np.corrcoef(yhat_test.ravel(), test_df['y'])[0, 1]**2 #compute R^2 between yhat_test and y_test    
    ax.scatter(yhat_test, y_test, color='C0', s=1, alpha=.1,
               label='test data')
    xlim = [min(yhat_test), max(yhat_test)]
    ax.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
    ax.set_xlabel('model prediction ($\hat{y}$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title(f'Standard metric of model performance:\n$R^2$={Rsq:.3}');
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'mavenn_measure_yhat.png'), facecolor='w', dpi=200)
    plt.close()
    
    
# plot mavenn y versus phi
def mavenn_phi(mavenn_model, test_df, saveDir):
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    phi_test = mavenn_model.x_to_phi(test_df['x']) #compute φ on test data
    phi_lim = [min(phi_test)-.5, max(phi_test)+.5] #set phi limit and create a grid in phi space
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
    yhat_grid = mavenn_model.phi_to_yhat(phi_grid) #compute yhat each phi grid point
    q = [0.025, 0.975] #compute 95% CI for each yhat
    yqs_grid = mavenn_model.yhat_to_yq(yhat_grid, q=q)
    ax.fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                   alpha=0.1, color='C1', lw=0, label='95% CI') #plot 95% confidence interval
    ax.plot(phi_grid, yhat_grid,
            linewidth=2, color='C1', label='nonlinearity') #plot GE nonlinearity
    y_test = test_df['y']
    ax.scatter(phi_test, y_test,
               color='C0', s=1, alpha=.1, label='test data',
               zorder=-100, rasterized=True) #plot scatter of φ and y values
    ax.set_xlim(phi_lim)
    ax.set_xlabel('latent phenotype ($\phi$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title('GE measurement process')
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(saveDir, 'mavenn_measure_phi.png'), facecolor='w', dpi=200)
    plt.close()
    

# plot first-order matrix and corresponding logo; for additive, single ISM, saliency, etc.
def matrix_and_logo(matrix, xticks, clim, label, L, P, alphabet, savePath, plot_matrix, plot_logo, ylabel):
    # create figure for first-order matrix
    if plot_matrix is True:
        plt.figure()
        ax = plt.gca()
        im = plt.pcolormesh(matrix,
                       edgecolors='k',
                       linewidth=1,
                       cmap='seismic')
        im.set_clim(vmin=-1.*clim, vmax=clim)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.0%', pad=0.15)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        B = ['A', 'C', 'G', 'T']
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(B, fontsize=12)
        ax.set_xticks(list(P))
        if xticks is not None:
            ax.set_xticklabels(xticks, fontsize=12)
        elif xticks is None:
            ax.set_xticklabels(list(L), fontsize=12)
        plt.colorbar(im, cmap='seismic', cax=cax)
        plt.tight_layout()
        plt.savefig(savePath + '.pdf', facecolor='w', dpi=600)
        plt.close()
        
    # create figure for corresponding logo
    if plot_logo is True: 
        if 0:
            fig, ax = plt.subplots(figsize=[7,2])
        else:
            fig, ax = plt.subplots(figsize=[7,1.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        logomaker.Logo(df=squid_utils.arr2pd(matrix.T, alphabet),
                        ax=ax,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(P)
        if xticks is not None:
            ax.set_xticklabels(xticks)
        elif xticks is None:
            ax.set_xticklabels(L)
        plt.tight_layout()
        plt.savefig(savePath + '_logo.pdf', facecolor='w', dpi=600)
        plt.close()

