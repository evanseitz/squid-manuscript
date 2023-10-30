# =============================================================================
# Script source: https://github.com/shtoneyan/gopher
# =============================================================================

import os, sys
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import tensorflow as tf
from tensorflow import keras

class Explainer():
    """wrapper class for attribution maps"""

    def __init__(self, model, example, class_index=None, func=tf.math.reduce_sum, binary=False):
        self.model = model
        self.example = example
        self.class_index = class_index
        self.func = func
        self.binary = binary

    def saliency_maps(self, X, batch_size=128):
        return function_batch(X, saliency_map, batch_size, model=self.model, example=self.example,
                              class_index=self.class_index, func=self.func,
                              binary=self.binary)


def function_batch(X, fun, batch_size=128, **kwargs):
    """ run a function in batches """

    dataset = tf.data.Dataset.from_tensor_slices(X)
    outputs = []
    for x in dataset.batch(batch_size):
        f = fun(x, **kwargs)
        outputs.append(f)
    return np.concatenate(outputs, axis=0)

def grad_times_input_to_df(x, grad, alphabet='ACGT'):
    """generate pandas dataframe for saliency plot
     based on grad x inputs """

    x_index = np.argmax(np.squeeze(x), axis=1)
    grad = np.squeeze(grad)
    L, A = grad.shape

    seq = ''
    saliency = np.zeros((L))
    for i in range(L):
        seq += alphabet[x_index[i]]
        saliency[i] = grad[i,x_index[i]]

    # create saliency matrix
    saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
    return saliency_df

@tf.function
def saliency_map(X, model, example, class_index=None, func=tf.math.reduce_sum, binary=False): #reduce_mean
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        if binary:
            outputs = model(X)[:, class_index]
        else:
            if example != 'CAGI5-GOPHER': #standard approach for a single class
                outputs = tf.math.reduce_sum(model(X)[:, :, class_index], axis=1)
            elif example == 'CAGI5-GOPHER':
                #outputs = tf.math.reduce_sum(model(X), axis=[1,2])
                outputs = tf.math.reduce_sum(tf.math.reduce_mean(model(X), axis=2))

    return tape.gradient(outputs, X)


# Enformer contribution code (below) edited from:
# https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb
class Enformer:
    def __init__(self, model):#tfhub_url):
       self._model = model#hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)

            prediction = tf.reduce_sum(
                target_mask[tf.newaxis] *
                self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)

        #return tf.reduce_sum(input_grad, axis=-1) #returns shape (393216,)
        return input_grad #returns shape (393216, 4)