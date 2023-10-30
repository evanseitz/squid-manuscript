# =============================================================================
# Code for generating deepLIFT attribution maps
# =============================================================================
# The following code is taken from the DeepSTARR GitHub repository script..
# ..DeepSTARR_nucl_contr_scores.py with slight modifications.
# This code relies on a forked version of SHAP maintained by the Kundaje Lab:
# https://github.com/AvantiShri/shap/blob/master/shap/explainers/deep/deep_tf.py 
# =============================================================================


import os, sys
sys.dont_write_bytecode = True
import numpy as np
import deeplift
from deeplift.dinuc_shuffle import dinuc_shuffle
pyDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pyDir, 'kundajelab_shap_29d2ffa/'))
import shap
from shap.explainers.deep.deep_tf import TFDeepExplainer

# deepExplainer functions
def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example,
                                seed=1234):
  assert len(list_containing_input_modes_for_an_example)==1
  onehot_seq = list_containing_input_modes_for_an_example[0]
  rng = np.random.RandomState(seed)
  dinuc_shuffle_n = 100
  to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(dinuc_shuffle_n)])
  return [to_return] #wrap in list for compatibility with multiple modes


# get hypothetical scores also
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp)==1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape)==2
    #At each position in the input sequence, we iterate over the one-hot encoding
    # possibilities (eg: for genomic sequence, this is ACGT i.e.
    # 1000, 0100, 0010 and 0001) and compute the hypothetical
    # difference-from-reference in each case. We then multiply the hypothetical
    # differences-from-reference with the multipliers to get the hypothetical contributions.
    #For each of the one-hot encoding possibilities,
    # the hypothetical contributions are then summed across the ACGT axis to estimate
    # the total hypothetical contribution of each position. This per-position hypothetical
    # contribution is then assigned ("projected") onto whichever base was present in the
    # hypothetical sequence.
    #The reason this is a fast estimate of what the importance scores *would* look
    # like if different bases were present in the underlying sequence is that
    # the multipliers are computed once using the original sequence, and are not
    # computed again for each hypothetical sequence.
    for i in range(orig_inp[0].shape[-1]):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:,i] = 1.0
        hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference*mult[0]
        projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1)
    return [np.mean(projected_hypothetical_contribs,axis=0)]


def deepExplainer(model, one_hot, class_output):
    # output layer
    if class_output==0: #'dev'
        out_layer=-2
    if class_output==1: #'hk'
        out_layer=-1

    explainer = TFDeepExplainer((model.layers[0].input, model.layers[out_layer].output),
        data=dinuc_shuffle_several_times, combine_mult_and_diffref=combine_mult_and_diffref)

    # running on all sequences
    shap_values_hypothetical = explainer.shap_values(one_hot)

    # normalising contribution scores
    # sum the deeplift importance scores across the ACGT axis (different nucleotides at the same position)
    # and “project” that summed importance onto whichever base is actually present at that position
    shap_values_contribution=shap_values_hypothetical[0]*one_hot

    return shap_values_hypothetical[0], shap_values_contribution