import os, sys, inspect
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd


def deep_sea(x, num_sim, avg_num_mut, alphabet, mode, mut_range=None):
    """
    In-silico deep-mutational sequencing

    x:              input sequence (one-hot encoding);
                        shape (L, 4)
    num_sim:        number of mutated sequences to generate;
                        positive integer
    avg_num_mut:    average number of mutations per sequence;
                        positive integer
    alphabet:       alphabet for sequence;
                        1D array: e.g., ['A', 'C', 'G', 'T']
    mut_range:      range of nucleotides to mutate on input sequence;
                        [0 <= int < L, 1 <= int <= L]; e.g., [1024, 1035]
                        if None, default to full range
    mode:           select type of mutagenesis to perform, either...
                    drawing the number of mutations per sequence from..
                    a Poisson or uniform distribution;
                        string: {'poisson', 'uniform'}
    """

    num_alphabet = len(alphabet)
    rng = np.random.default_rng()

    L, A = x.shape

    # trim wild-type to mutation range (if chosen) to avoid memory failure when using very large sequences
    # (removed portions of sequence are appended back on in a subsequent function for model predictions)
    if mut_range is not None:
        x = x[mut_range[0]:mut_range[1],:]
        L = mut_range[1] - mut_range[0]
    elif mut_range is None:
        mut_range = [0, L] #full length of sequence

    # get shapes
    ###if mut_range is None:
        ###mut_range = [0, L] #full length of sequence
    ###L_mut = mut_range[1] - mut_range[0]
    
    wt_argmax = np.argmax(x, axis=1) #get indices of nucleotides
    one_hots = np.zeros((num_sim, L, A)) #generate one-hot mutations
    
    # generate number of mutations
    if mode == 'poisson':
        num_muts = np.random.poisson(avg_num_mut, (num_sim, 1))[:,0]
        num_muts = np.clip(num_muts, 0, L)###L_mut)
        for i, num_mut in enumerate(num_muts):
            # sample positions to mutate
            ###mut_index = rng.choice(range(mut_range[0], mut_range[1]), num_mut, replace=False)
            mut_index = rng.choice(range(0, L), num_mut, replace=False)
            # generate index of mutation
            mut = rng.choice(range(1, num_alphabet), (num_mut))
            # loop through sequence and add mutation index (note: up to 3 is added which does not map to [0,3] alphabet)
            mut_argmax = np.copy(wt_argmax)
            for j,m in zip(mut_index, mut):
                mut_argmax[j] += m
            # wrap non-sensical indices back to alphabet -- effectively makes it random mutation
            seq_index = np.mod(mut_argmax, 4)
            # create one-hot representation
            one_hots[i,:,:] = np.eye(num_alphabet)[seq_index]
        
    elif mode == 'uniform':
        num_mut = int(avg_num_mut)
        num_mut = np.clip(num_mut, 0, L)###L_mut)
        # generate index of mutation
        mut = rng.choice(range(1, num_alphabet), (num_sim, num_mut))
        for i in range(num_sim):
            # sample positions to mutate
            mut_index = rng.choice(range(mut_range[0], mut_range[1]), num_mut, replace=False)
            # generate mutation index vector
            z = np.zeros((L), dtype=int)
            z[mut_index] = mut[i,:]
            # wrap non-sensical indices back to alphabet -- effectively makes it random mutation
            seq_index = np.mod(wt_argmax + z, 4)
            # create one-hot representation
            one_hots[i,:,:] = np.eye(num_alphabet)[seq_index]
            
    else:
        print('Unknown definition for Mode: "%s"' % mode)
        return None
  
    return one_hots


def calamari(one_hots, x, model, class_num, alphabet, GPU, example, get_prediction, unwrap_prediction, compress_prediction, unwrap_pred_wt, pred_transform, pred_trans_delimit, delimit_start, delimit_stop, bin_res, output_skip, max_in_mem, saveDir, squid_utils, mut_range=None):
    """
    Generate in-silico MAVE dataset
    
    one_hots:   matrix of N one-hots generated from function deep_sea()
                    shape (N, L, 4)
    x:          one-hot encoding of original (wild-type) sequence;
                    shape (L, 4)
    model:      trained model
    class_num:  index of prediction class read from model;
                    positive integer
    alphabet:   alphabet for sequence;
                    1D array: e.g., ['A', 'C', 'G', 'T']
    GPU:        running GPUs (True) or CPUs (False);
                    ..translating entire matrix at once is very slow on CPUs
    """

    N = one_hots.shape[0]
    L = one_hots.shape[1]
    mave_df = pd.DataFrame(columns = ['y', 'x'], index=range(N))

    if mut_range is not None:
        x1 = x[:mut_range[0], :]
        x2 = x[mut_range[1]:, :]
    
    # translate matrix of one-hots into sequence dataframe
    if GPU is True: #entire matrix at once (~twice as fast as standard approach if running on GPUs)
        seq_index_all = np.argmax(one_hots, axis=-1)
        num2alpha = dict(zip(range(0, len(alphabet)), alphabet))
        seq_vector_all = np.vectorize(num2alpha.get)(seq_index_all)
        seq_list_all = seq_vector_all.tolist()
        dictionary_list = []
        for i in range(0, N, 1):
            dictionary_data = {0: ''.join(seq_list_all[i])}
            dictionary_list.append(dictionary_data)
        mave_df['x'] = pd.DataFrame.from_dict(dictionary_list)
    
    elif GPU is False:
        alphabet_dict = {}
        idx = 0
        for i in range(len(alphabet)):
            alphabet_dict[i] = alphabet[i]
            idx += 1
        for i in range(N): #standard approach
            seq_index = np.argmax(one_hots[i,:,:], axis=1)
            seq = []
            for s in seq_index:
                seq.append(alphabet_dict[s])
            seq = ''.join(seq)
            mave_df.at[i, 'x'] = seq


    if max_in_mem is not False: #batch prediction mode
        """
        WARNING: for large models like ENFORMER, the created memmap can quickly exceed..
                ..local disk space; e.g., for N=100000, and ~20 mb per ENFORMER each..
                ..shape=(896,5313) prediction, the resulting memmap will be 2 TB!
                As a precaution, the 'default' option below is set so as to not save to disk
        """
        default = True

        if mut_range is None:
            single_pred = get_prediction(np.expand_dims(one_hots[0], 0), example, model)
        else: #fill back in previously-pruned first and last sequence segments
            single_pred = get_prediction(np.expand_dims(np.vstack((x1, one_hots[0], x2)),0), example, model)

        if default is False:
            memmap_out = os.path.join(saveDir, 'mave_preds_raw.npy')
            pred_all = np.memmap(memmap_out, mode='w+', dtype='float32', 
                                 shape=(N, single_pred.shape[1], single_pred.shape[2]))

        def partition(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]
        batches = list(partition(np.arange(N), max_in_mem))

        b_total = 0
        for batch_idx in range(len(batches)):
            print('Batch: %s of %s' % (batch_idx+1, len(batches)))

            if mut_range is None:
                preds = get_prediction(one_hots[batches[batch_idx]], example, model)
            else:
                # fill back in previously-pruned first and last sequence segments
                x1_dup = np.broadcast_to(x1,(len(batches[batch_idx]),)+x1.shape)
                x2_dup = np.broadcast_to(x2,(len(batches[batch_idx]),)+x2.shape)
                one_hots_full = np.hstack((x1_dup, one_hots[batches[batch_idx]], x2_dup))
                preds = get_prediction(one_hots_full, example, model)

            if default is False:
                pred_all[batches[batch_idx]] = preds
            else:
                for n in range(len(batches[batch_idx])):
                    if pred_transform == 'pca':
                        mave_df.at[b_total, 'y'] = unwrap_prediction(preds, class_num, n, example, pred_transform)
                    else:
                        unwrap = unwrap_prediction(preds, class_num, n, example, pred_transform)
                        mave_df.at[b_total, 'y'] = compress_prediction(unwrap, pred_transform=pred_transform, pred_trans_delimit=pred_trans_delimit,
                                                                    delimit_start=(delimit_start/float(bin_res))-output_skip, delimit_stop=(delimit_stop/float(bin_res))-output_skip)

                    b_total += 1

        if default is False:
            for n in range(N):
                if pred_transform == 'pca':
                    mave_df.at[n, 'y'] = unwrap_prediction(pred_all, class_num, n, example, pred_transform)
                else:
                    unwrap = unwrap_prediction(pred_all, class_num, n, example, pred_transform)
                    mave_df.at[n, 'y'] = compress_prediction(unwrap, pred_transform=pred_transform, pred_trans_delimit=pred_trans_delimit,
                                                                delimit_start=(delimit_start/float(bin_res))-output_skip, delimit_stop=(delimit_stop/float(bin_res))-output_skip)

            del pred_all
            if 1:
                os.remove(memmap_out)

    else: #no batch predictions
        if mut_range is None:
            pred_all = get_prediction(one_hots, example, model) #get predictions from model in one batch
        else:
            # fill back in previously-pruned first and last sequence segments
            x1_dup = np.broadcast_to(x1,(N,)+x1.shape)
            x2_dup = np.broadcast_to(x2,(N,)+x2.shape)
            one_hots_full = np.hstack((x1_dup, one_hots, x2_dup))
            pred_all = get_prediction(one_hots_full, example, model)

        for n in range(N):
            if pred_transform == 'pca':
                mave_df.at[n, 'y'] = unwrap_prediction(pred_all, class_num, n, example, pred_transform)
            else:
                unwrap = unwrap_prediction(pred_all, class_num, n, example, pred_transform)
                mave_df.at[n, 'y'] = compress_prediction(unwrap, pred_transform=pred_transform, pred_trans_delimit=pred_trans_delimit,
                                                            delimit_start=(delimit_start/float(bin_res))-output_skip, delimit_stop=(delimit_stop/float(bin_res))-output_skip)


        del pred_all

    
    # prepare dataset for MAVE-NN
    mave_df['set'] = np.random.choice(a=['training','test','validation'],
                                       p=[.6,.2,.2],
                                       size=len(mave_df))
    new_cols = ['set'] + list(mave_df.columns[0:-2]) + ['x']
    mave_df = mave_df[new_cols]


    if pred_transform == 'pca':
        mave_df, pred_wt = squid_utils.pca(mave_df, unwrap_pred_wt, saveDir, save_name='mave')
    else:
        pred_wt = compress_prediction(unwrap_pred_wt, pred_transform=pred_transform, pred_trans_delimit=pred_trans_delimit,
                                      delimit_start=(delimit_start/float(bin_res))-output_skip, delimit_stop=(delimit_stop/float(bin_res))-output_skip)

    return (mave_df, pred_wt)

