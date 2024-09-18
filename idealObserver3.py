#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:46:15 2022

@author: Alexander Paunov, Florent Meyniel
"""

import numpy as np
import pandas as pd
from scipy import stats as sp

# Add selected derived variables to ideal observer
def io_with_derivations(sequence, vol, sd=(10, 20),
                        which_variables=['all'], as_predictors=True,
                        reward_levels=(30, 50, 70),
                        reward_range=(1, 100), window_size=2, prior_init=None):

    io = ideal_observer(sequence, vol, sd, reward_levels=(
        30, 50, 70), prior_init=prior_init)

    sequence = {'options': io['options'],
                'outcome_SD_mdl': io['outcome_SD_mdl']}

    if as_predictors:
        id_dist = 'prior'
        uu_return_prior = True
    else:
        id_dist = 'posterior'
        uu_return_prior = False


    if which_variables == ['all']:
        # DROPPED FOR NOW: 'knowledge_gradient'
        which_variables = ['MAP_reward',
                           'expected_reward',
                           'expected_uncertainty',
                           'unexpected_uncertainty',
                           'prediction_error',
                           'unsigned_prediction_error',
                           'prediction_error_MAP',
                           'unsigned_prediction_error_MAP',
                           'feedback_surprise',
                           'signed_feedback_surprise',
                           'expected_discrete_reward',
                           'expected_outcome_uncertainty',
                           'estimation_confidence',
                           'entropy']

    vars_requiring_reward_prob = ['prior_reward_probability',
                                  'posterior_reward_probability',
                                  'expected_discrete_reward',
                                  'expected_outcome_uncertainty']
    vars_requiring_prior_reward_prob = ['feedback_surprise',
                                  'signed_feedback_surprise']

    get_reward_prob = np.any([var in vars_requiring_reward_prob
                              for var in which_variables])
    get_prior_reward_prob = id_dist != "prior"
    if not get_prior_reward_prob:
        get_prior_reward_prob = np.any([var in vars_requiring_prior_reward_prob
                                        for var in which_variables])
    if get_reward_prob:
        id_reward_prob = id_dist + '_reward_probability'
        io[id_reward_prob] = \
            reward_probability(io[id_dist], io['outcome_SD_mdl'],
                               io['reward_levels'], reward_range)
            
    if get_prior_reward_prob:
        id_reward_prob = 'prior_reward_probability'
        io[id_reward_prob] = \
            reward_probability(io['prior'], io['outcome_SD_mdl'],
                               io['reward_levels'], reward_range)

    for v in which_variables:
        if v == 'MAP_reward':
            io[v] = MAP_reward(io[id_dist], io['reward_levels'])
        elif v == 'expected_reward':
            io[v] = expected_reward(io[id_dist], io['reward_levels'])
        elif v == 'expected_uncertainty':
            io[v] = expected_uncertainty(io[id_dist])
        elif v == 'unexpected_uncertainty':
            io[v] = unexpected_uncertainty(io['posterior'],  # Note: not dist
                                           io['volatility'],
                                           io['reward_levels'],
                                           sequence,
                                           window_size=window_size,
                                           return_prior=uu_return_prior)
             
        elif v == 'prediction_error':
            io[v] = prediction_error(sequence,
                                     io["prior"],
                                     # io[id_dist],  #NB: The next ones up to signed_surprise are only correct with prior!
                                     io['reward_levels'])
        elif v == 'unsigned_prediction_error':
            io[v] = absolute_prediction_error(sequence,
                                              io['prior'],
                                              io['reward_levels'])
        elif v == 'prediction_error_MAP':
            io[v] = prediction_error_MAP(sequence,
                                         io['prior'],
                                         io['reward_levels'])
        elif v == 'unsigned_prediction_error_MAP':
            io[v] = absolute_prediction_error_MAP(sequence,
                                                  io['prior'],
                                                  io['reward_levels'])
        elif v == 'feedback_surprise':
            io[v] = outcome_surprise(
                # io[id_reward_prob], 
                io['prior_reward_probability'], 
                io['options'])
        elif v == 'signed_feedback_surprise':
            io[v] = signed_outcome_surprise(io['prior_reward_probability'], io['prior'],
                                            sequence, io['reward_levels'])
        elif v == 'expected_discrete_reward':
            io[v] = expected_reward_discrete(io[id_reward_prob])
        elif v == 'expected_outcome_uncertainty':
            io[v] = expected_uncertainty_outcome(io[id_reward_prob])
        elif v == 'estimation_confidence':
            io[v] = estimation_confidence(io[id_dist])
        elif v == 'entropy':
            io[v] = entropy (io['prior'])

    return io

def ideal_observer(sequence, vol, sd, reward_levels=(30, 50, 70), prior_init=None):
    """Get ideal observer variables.

    Compute the ideal observer inference of the generative reward levels
    given the sequence of observations, the assumed volatility (*vol*) and
    list of possible *reward_levels*, and *rewards_raange*

    *sequence* is a dictionary with the following fields:
        - 'options': dictionary of outcomes (provided as numpy array, nan
           indicate no observation) for the different options (provided as the
           key name)
        - sequence['outcome_SD_gen']: a numpy array of the generative outcome SD on each
          trial

    *rewards range* is a tuple of min and max rewards

    The *window_size* is the number of trials taken into account to estimate
    unexpected uncertainty.

    NB: default past_window_size of 2 matches results with past_window_size of
    3 in matlab
    """
    # define transition probability matrix
    n_levels = len(reward_levels)
    trans = (1-vol)*np.eye(n_levels) + \
        vol*(1-np.eye(n_levels))*(1/(n_levels-1))

    # define flat prior
    if prior_init is None:
        prior_init = {}
        for option_name in sequence['options'].keys():
            prior_init[option_name] = np.ones(n_levels) / n_levels

    # Compute inference
    predictive = {option_name: [] for option_name in sequence['options']}
    posterior = {option_name: [] for option_name in sequence['options']}
    prior = {option_name: [] for option_name in sequence['options']}
    predictive_prior = {option_name: [] for option_name in sequence['options']}

    
    if type(sequence['outcome_SD_gen']) == np.int64:
        n_trials = 1
    else:
        n_trials = len(sequence['outcome_SD_gen'])
        
    if n_trials == 96:  # NB: THIS BYPASSES USING ESTIMATED SDS FOR PRM RECOVERY, FINDING REWARD-MAXIMIZING MODEL
        sequence['outcome_SD_mdl'] = np.vstack(
            [sd] * int(n_trials/2)).ravel('f')

        lowsd_is_first = sequence['outcome_SD_gen'][0] < sequence['outcome_SD_gen'][-1]
        if not lowsd_is_first:
            sequence['outcome_SD_mdl'] = np.flip(sequence['outcome_SD_mdl'])
    elif n_trials == 1:
        sequence['outcome_SD_mdl'] = np.array([sequence['outcome_SD_gen']])
        sequence['options'] = {arm: np.array([a]) for arm, a in sequence['options'].items()}
    else:
        sequence['outcome_SD_mdl'] = sequence['outcome_SD_gen']

    for option_name, option_outcomes in sequence['options'].items():
        for (outcome, SD) in zip(option_outcomes, sequence['outcome_SD_mdl']):
            # get outcome likelihoods
            lik = outcome_likelihood(outcome, reward_levels, SD)

            # update estimate of reward distribution
            if len(posterior[option_name]) == 0:
                unscaled_post = lik * (trans @ prior_init[option_name])
            else:
                unscaled_post = lik * (trans @ posterior[option_name][-1])

            # normalize reward distributions to probabilities
            posterior[option_name].append(unscaled_post / sum(unscaled_post))

            # turn probabilities to predictions
            predictive[option_name].append(trans @ posterior[option_name][-1])

        posterior[option_name] = np.vstack(posterior[option_name])
        predictive[option_name] = np.vstack(predictive[option_name])

    for option_name in sequence['options']:
        prior[option_name] = np.vstack(
            (prior_init[option_name], posterior[option_name][:-1, :]))
        predictive_prior[option_name] = np.vstack(
            (prior_init[option_name], predictive[option_name][:-1, :]))

    return {'posterior': posterior,
            'predictive': predictive,
            'prior': prior,
            'predictive_prior': predictive_prior,
            'volatility': vol,
            'reward_levels': reward_levels,
            'options': sequence['options'],
            'outcome_SD_gen': sequence['outcome_SD_gen'],
            'outcome_SD_mdl': sequence['outcome_SD_mdl'],
            'choices': get_choices(sequence)}


def outcome_likelihood(outcome, reward_levels, SD):
    """Compute likelihood.

    Joint likelihood of *outcome*(s) given the possible *reward_levels* and
    the assumed *SD* at each observation.
    Outcome can be a single observation, or an array of observations (in that
    case, SD should be an array too, of the same size as *outcome*.
    """
    n_levels = len(reward_levels)

    def single_outcome_likelihood(single_outcome, reward_levels, single_SD):
        return np.ones(n_levels) if np.isnan(single_outcome) else\
            sp.norm.pdf(single_outcome, reward_levels, single_SD)

    one_trial = False
    if type(outcome) in [float, int]:
        one_trial = True
    if not one_trial:
        if outcome.shape == ():
            one_trial = True
    
    if one_trial:
        lik = single_outcome_likelihood(outcome, reward_levels, SD)
    else:
        # get joint likelihood of outcomes
        lik = np.prod([single_outcome_likelihood(
            single_outcome, reward_levels, single_SD)
            for (single_outcome, single_SD) in zip(outcome, SD)], axis=0)
    return lik

# For adding choices to IO outputs
def get_choices(sequence):
    if len(sequence["options"]["A"]) == 1:
        is_chosen = []
        for arm in sequence["options"].keys():
            is_chosen.append(sequence["options"][arm])
        is_chosen = ~np.isnan(is_chosen)
        choices = is_chosen[1].astype(float)
    else:
        is_chosen = ~np.isnan(pd.DataFrame(sequence['options'])).values
        missing = ~np.any(is_chosen, axis=1)
        choices = is_chosen[:, 1].astype(float)
        choices[missing] = np.nan

    return choices

# Derived variables
def MAP_reward(prior, reward_levels):
    """Get maximum a posteriori reward.
    Note: currently, in case of 2-way ties, defaults to 1st index"""
    RMAP = {}
    for option_name in prior.keys():
        RMAP[option_name] = []
        for it in range(prior[option_name].shape[0]):
            if np.all(prior[option_name][it, :] == 1/(len(reward_levels))):
                RMAP[option_name].append(np.mean(reward_levels))
            else:
                RMAP[option_name].append(
                    reward_levels[np.argmax(prior[option_name][it, :])])
        RMAP[option_name] = np.asarray(RMAP[option_name]).astype(int)

    return RMAP

    # return {option_name: [reward_levels[loc] for loc in
    #                       np.argmax(distribution, axis=1)]
    #         for (option_name, distribution) in prior.items()}


def expected_reward(prior, reward_levels=(30, 50, 70)):
    """Get expected reward."""
    return {option_name: option_post @ reward_levels
            for option_name, option_post in prior.items()}


def expected_uncertainty(prior):
    """Get expected uncertainty.

    Compute the expected uncertainty of the *predictive_distribution*
    (dictionary of numpy arrays of shape [trials, reward levels]), as the
    probatility of the Maximum A Posterior value.
    """
    return {option_name: 1 - np.amax(option_post, 1)
            for option_name, option_post in prior.items()}


def estimation_confidence(prior):
    """ Get confidence: 1-EU """
    uncertainty = expected_uncertainty(prior)
    confidence = {option_name: (1 - eu) for option_name, eu in uncertainty.items()}
    
    return confidence

def posterior_no_change(past_trial, option_posteriors, option_outcomes,
                        SDs, vol, reward_levels, trans_change):
    """Get posterior with no change point.

    Compute the posterior probability of no change point on a given past
    trial, assuming that there is no change point after this past trial
    until the current trial included.
    Note that past_trial=0 is the current trial, past_trial=1 is the
    previous trial, etc.
    """
    # use Python backward indices
    index_current, index_previous = 1, 2

    # get (joint) likelihood of observation(s) given each possible fixed
    # reward mean and SD
    outcome_lik = outcome_likelihood(
        option_outcomes[-(index_current+past_trial):],
        reward_levels, SDs[-(index_current+past_trial):])

    # get unscaled posterior probability of a change point (resp. no change
    # point) on a given past_trial, assuming no change point after this
    # past_trial
    prob_no_change = (1-vol)**(1+past_trial) * (
        option_posteriors[-(index_previous+past_trial)] @ outcome_lik)

    prob_change = vol*(1-vol)**past_trial * (
        option_posteriors[-(index_previous+past_trial)] @
        (trans_change @ outcome_lik))

    # return the scaled posterior prob of no change on the given past_trial
    return prob_no_change / (prob_no_change + prob_change)

def unexpected_uncertainty(posterior, vol,
                           reward_levels, sequence,
                           window_size, return_prior=False):
    """Get unexpected uncertainty.

    Compute the posterior probability that there has been at least one change
    point in the recent past trials (within a window of window_size trials,
    which includes the current trial and window_size-1 past trials).
    If return_prior=True, the prior is return instead of the posterior.
    """
    n_levels = len(reward_levels)
    trans_change = (1-np.eye(n_levels))*(1/(n_levels-1))
    UU_this_past_window_size = {option_name: []
                                for option_name in sequence['options'].keys()}
    for option_name in sequence['options'].keys():
        # initialize:
        for t in range(window_size):
            UU_this_past_window_size[option_name].append(
                1 - (1-vol) ** window_size)

        for t in range(window_size, len(posterior[option_name])):
            # use a recursive formula for computing unexpected uncertainty
            for past_trial in range(window_size):
                if past_trial == 0:
                    # this is for the current trial
                    # NB: x[:t+1] returns the values x[0], ... x[t]
                    UU = 1-posterior_no_change(
                        past_trial,
                        posterior[option_name][:t+1],
                        sequence['options'][option_name][:t+1],
                        sequence['outcome_SD_mdl'][:t+1],
                        vol, reward_levels, trans_change)
                else:
                    # this is for trials before the current trial
                    UU = 1-(1-UU)*posterior_no_change(
                        past_trial,
                        posterior[option_name][:t+1],
                        sequence['options'][option_name][:t+1],
                        sequence['outcome_SD_mdl'][:t+1],
                        vol, reward_levels, trans_change)
            UU_this_past_window_size[option_name].append(UU)

        if return_prior:
            # add the prior and shift
            UU_this_past_window_size[option_name] = \
                [1 - ((1-vol) ** window_size)] + \
                UU_this_past_window_size[option_name][:-1]

        # From list to numpy array
        UU_this_past_window_size[option_name] = \
            np.asarray(UU_this_past_window_size[option_name])

    return UU_this_past_window_size

def prediction_error(sequence, prior, reward_levels):
    """Get prediction error from ER.

    Difference between observed and expected reward.
    Defaults to nan for unobserved arm.
    """
    predicted_reward = expected_reward(prior, reward_levels)

    PE = {option_name: [] for option_name in predicted_reward}
    for option_name in PE:
        PE[option_name] = sequence['options'][option_name] - \
            predicted_reward[option_name]

    return PE


def prediction_error_MAP(sequence, prior, reward_levels):
    """Get prediction error from MAP.

    Difference between observed and *MAP* reward.
    Defaults to nan for unobserved arm.
    """
    predicted_reward = MAP_reward(prior, reward_levels)

    PE = {option_name: [] for option_name in predicted_reward}
    for option_name in PE:
        PE[option_name] = sequence['options'][option_name] - \
            predicted_reward[option_name]

    return PE

def absolute_prediction_error(sequence, prior, reward_levels):
    """Get absolute prediction error from ER. 

    Difference between observed and expected reward.
    Defaults to nan for unobserved arm.
    """
    PE = prediction_error(sequence, prior, reward_levels)
    abs_PE = {option_name: [] for option_name in PE}
    for option_name in PE:
        abs_PE[option_name] = np.abs(PE[option_name])

    return abs_PE

def absolute_prediction_error_MAP(sequence, prior, reward_levels):
    """Get absolute prediction error from MAP. 

    Difference between observed and *MAP* reward.
    Defaults to nan for unobserved arm.
    """
    PE = prediction_error_MAP(sequence, prior, reward_levels)
    abs_PE = {option_name: [] for option_name in PE}
    for option_name in PE:
        abs_PE[option_name] = np.abs(PE[option_name])

    return abs_PE

def reward_probability(prior, outcome_SD, reward_levels, reward_range):
    """Get probability of each reward from the probabilities of each reward
    level, at a given outcome SD."""
    n_reward_levels = len(reward_levels)
    n_trials = len(outcome_SD)

    rewards = np.arange(reward_range[0], reward_range[1] + 1)
    n_rewards = len(rewards)

    p_reward = {arm: [] for arm in prior}
    for arm in prior:
        p_reward[arm] = np.nan * np.ones((n_trials, n_rewards))
        for t in range(n_trials):
            p_reward_per_level = np.nan * np.ones((n_reward_levels, n_rewards))
            for ir, r in enumerate(reward_levels):
                # Get gaussian probability density around given reward level
                # with outcome SD variance
                p_reward_per_level[ir, :] = \
                    sp.norm.pdf(rewards, loc=r, scale=outcome_SD[t])
                # Assign total probability mass at this reward level from IO
                # posterior per reward level
                p_reward_per_level[ir, :] = (p_reward_per_level[ir, :] /
                                             np.sum(p_reward_per_level[ir, :])
                                             ) * prior[arm][t, ir]

                # Get posterior per reward (sum across reward levels)
                p_reward[arm][t, :] = np.sum(p_reward_per_level, axis=0)

    return p_reward


def outcome_surprise(p_reward, observed_reward):
    "Negative log probability of the observed reward"
    n_trials = pd.DataFrame(observed_reward).shape[0]

    surprise = {arm: [] for arm in p_reward}
    for arm in p_reward:
        surprise[arm] = np.nan * np.ones(n_trials)
        for itrial, r in enumerate(observed_reward[arm]):
            if ~np.isnan(r):
                p_observed_reward = p_reward[arm][itrial, int(r) - 1]
                surprise[arm][itrial] = -np.log(p_observed_reward)

    return surprise


def signed_outcome_surprise(p_reward, prior, sequence, reward_levels):

    surprise = outcome_surprise(p_reward, sequence['options'])
    PE = prediction_error(sequence, prior, reward_levels)
    signed_surprise = {option: [] for option in surprise}
    for option in signed_surprise:
        signs = np.sign(PE[option])
        signs[signs == 0] = 1
        signed_surprise[option] = surprise[option] * signs

    return signed_surprise

def expected_reward_discrete(p_reward):
    """The MAP discrete reward, rather than obtained from sum of weighted
    reward levels (very similar)."""
    expected_discrete_reward = {arm: [] for arm in p_reward}
    for arm in p_reward:
        expected_discrete_reward[arm] = np.argmax(p_reward[arm], axis=1) + 1

    return expected_discrete_reward


def expected_uncertainty_outcome(p_reward):
    "The expected uncertainty of the outcome as opposed to the reward level"
    expected_outcome_uncertainty = {arm: [] for arm in p_reward}
    for arm in p_reward:
        expected_outcome_uncertainty[arm] = 1 - np.max(p_reward[arm], axis=1)

    return expected_outcome_uncertainty

def entropy(prior):
    "Entropy of the prior ditribution"

    entropy = {arm: [] for arm in prior}
    for arm in prior:
       entropy[arm] = -np.sum(prior[arm] * np.log(prior[arm]), axis=1)

    return entropy



