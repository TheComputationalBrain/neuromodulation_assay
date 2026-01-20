#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains utility functions for the IO estimates.

@authors: Alice Hodapp, Tiffany Bounmy, Cédric Foucault, Florent Meyniel
"""

import numpy as np
from scipy.stats import dirichlet
import sys 
import os

root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(root)
root_model = os.path.join(project_root, "TransitionProbModel")
if root_model not in sys.path:
    sys.path.append(root_model)

from TransitionProbModel.MarkovModel_Python import IdealObserver as IO

def get_conditional_value(val, seq):
    """
    This function allows to select the conditional value on the current
    observation for the transition probabilities case.
    val is a dictionnary corresponding to a conditional posterior value.
    The key is a tuple indicating the observation the value is conditioned upon.
    E.g. val[(0, 1)][k] is the posterior value of "1 given 0", estimated with seq[k] included.
    """

    seqL = seq.size
    conditional_val = np.zeros(seqL)

    # Retrieve the correct conditional value according to the sequence
    for k in range(seqL):
        conditional_val[k] = val[(seq[k], 1)][k]

    return conditional_val

def compute_p1_dist_pre(dist, seq, res, prior_init):
    """
    This function creates the prior distribution according to the shown sequence.

    Parameters
    ----------
    dist: dictionnary of numpy.ndarray corresponding to conditional posterior distributions
          the key is a tuple indicating the observation the distribution is conditionned upon.
          e.g. dist[(0, 1)][k] is the posterior probability of "1 given 0", estimated with seq[k]
          included.
    seq: array, sequence of observations
    res: str, resolution of the distribution

    Returns
    -------
    p1_dist_pre: numpy.ndarray, final disitribution

    """
    seqL = seq.size
    p1_dist_pre = np.zeros((res, seqL))

    # Retrieve the correct distribution according to the sequence
    p1_dist_pre[:, 0] = prior_init 
    for k in np.arange(1,seqL):
        p1_dist_pre[:, k] = dist[(seq[k], 1)][:,k-1]

    return p1_dist_pre

def compute_p1_dist(dist, seq, res):
    """
    This function creates the p1 distribution according to the shown sequence.

    Parameters
    ----------
    dist: dictionnary of numpy.ndarray corresponding to conditional posterior distributions
          the key is a tuple indicating the observation the distribution is conditionned upon.
          e.g. dist[(0, 1)][k] is the posterior probability of "1 given 0", estimated with seq[k]
          included.
    seq: array, sequence of observations
    res: str, resolution of the distribution

    Returns
    -------
    p1_dist: numpy.ndarray, final disitribution

    """

    seqL = seq.size
    p1_dist = np.zeros((res, seqL))

    # Retrieve the correct distribution according to the sequence
    for k in range(seqL):
        p1_dist[:, k] = dist[(seq[k], 1)][:,k]

    return p1_dist

def compute_entropy(p, base=None):
    """Defines the information entropy according to p"""
    if base == 2:
        H = -(p*np.log2(p) + (1-p)*np.log2(1-p))
    else:
        H = - (p * np.log(p) + (1-p) * np.log(1-p))

    return H

def kl_divergence(p1_pre, p1):
    p1_pre = np.array(p1_pre, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p1_pre = np.clip(p1_pre, 1e-10, None)  # Avoid zero probabilities in p
    p1 = np.clip(p1, 1e-10, None)  # Avoid zero probabilities in q
    return np.sum(p1_pre * np.log(p1_pre / p1))

def get_post_inference(seq, seq_type, options):
    """
    This function computes the posterior values for the distribution, the mean and
    the standard deviation given the sequence.
    """
    #initial prior
    prior_init = np.ones(options['resol']) / options['resol']

    if seq_type == 'bernoulli':
        out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

        p1_dist_array = out_hmm[(1,)]['dist']
        p1_dist_array_pre = np.zeros((options['resol'],seq.size)) 
        p1_dist_array_pre[:,0] = prior_init
        p1_dist_array_pre[:,1:] = out_hmm[(1,)]['dist'][:,:-1]
        p1_mean_array = out_hmm[(1,)]['mean']
        p1_sd_array = out_hmm[(1,)]['SD']

    if seq_type == 'transition':
        out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)

        dist = {(0, 1): out_hmm[(0, 1)]['dist'], (1, 1): out_hmm[(1, 1)]['dist']}
        p1_dist_array = compute_p1_dist(dist, seq, options['resol'])
        p1_dist_array_pre = compute_p1_dist_pre(dist, seq, options['resol'], prior_init)

        val = {(0, 1): out_hmm[(0, 1)]['mean'], (1, 1): out_hmm[(1, 1)]['mean']}
        p1_mean_array = get_conditional_value(val, seq)

        val = {(0, 1): out_hmm[(0, 1)]['SD'], (1, 1): out_hmm[(1, 1)]['SD']}
        p1_sd_array = get_conditional_value(val, seq)

    update = np.zeros(seq.size)
    for i in range(seq.size):
        update[i] = kl_divergence(p1_dist_array_pre[:, i], p1_dist_array[:, i])

    io_inference = {'seq': seq,
                    'p1_dist_array': p1_dist_array,
                    'p1_mean_array': p1_mean_array,
                    'p1_sd_array': p1_sd_array,
                    'prior_prediction_SDp0': out_hmm['prior_prediction_SDp0'],
                    'p1': 1 - out_hmm['prior_prediction_p0'],
                    'confidence_pre': -np.log(out_hmm['prior_prediction_SDp0']),
                    'confidence_post': -np.log(p1_sd_array),
                    'surprise': out_hmm['surprise'],
                    'entropy': compute_entropy(1 - out_hmm['prior_prediction_p0']),
                    'update': update}

    # fix the surprise on the first trial: force it to 1 bit
    io_inference['surprise'][0] = 1

    return io_inference