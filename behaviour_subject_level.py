#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Audrey Mazancieux, Tiffany Bounmy, Florent Meyniel

This script performs the subject-level analysis for the probability learning task of 
the NACONF dataset.
The output of this script is used by another script to do the group-level analysis.

The code has been adapted to work with the Neuromod project
"""

# Import useful libraries
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from params_and_paths import Paths, Params

paths = Paths()
params = Params()

def compute_entropy(p, base=2):
    """Defines the information entropy according to p"""
    if base == 2:
        H = -(p*np.log2(p) + (1-p)*np.log2(1-p))
    else:
        H = - (p * np.log(p) + (1-p) * np.log(1-p))

    return H


# Set dataset dir and parameters
SUBJECTS = params.naconf_behav_subj
ROOT_DIR = os.path.join(paths.root_dir,paths.data_dir)
BEHAV_DIR = os.path.join(ROOT_DIR,'behaviour_data')
OUT_DIR = '/home_local/alice_hodapp/NeuroModAssay/domain_general/behavior'
SAVE = True

all_sub_prob = {sub: None for sub in SUBJECTS}
all_sub_conf = {sub: None for sub in SUBJECTS}
all_stim_onsets = {sub: [] for sub in SUBJECTS}
all_IO_prob = {sub: [] for sub in SUBJECTS}
all_IO_conf = {sub: [] for sub in SUBJECTS}
all_IO_surp = {sub: [] for sub in SUBJECTS}
all_IO_entr = {sub: [] for sub in SUBJECTS}
all_sub_entr = {sub: [] for sub in SUBJECTS}
all_sub_res_conf = {sub: [] for sub in SUBJECTS}
all_not_missed_trials = {sub: [] for sub in SUBJECTS}

corr_dict = {'r_prob': {}, 'r_conf': {}, 'res_r_conf': {}}

for subject_id in SUBJECTS:
    print('SUBJECT', subject_id)
    # path to directories
    subj_behav_dir = os.path.join(ROOT_DIR, BEHAV_DIR, f'sub-{subject_id:02d}')

    # get task informations
    with open(os.path.join(subj_behav_dir,
                           f'experiment_info_sub-{subject_id:02d}.pickle'), 'rb') as f:
        exp = pickle.load(f)

    all_sub_prob[subject_id] = np.concatenate(
        [session['sub_prob'] for session in exp])
    all_sub_conf[subject_id] = np.concatenate(
        [session['sub_conf'] for session in exp])

    # get IO inference when questions
    io_prob = []
    io_conf = []
    io_surp = []
    io_entropy = []
    is_not_missed = []
    for sess in exp:
        # get IO estimated in response to the question
        io_prob.append(sess['p1_mean_array'][sess['stim_q']])
        io_conf.append(sess['confidence_post'][sess['stim_q']])
        io_surp.append(sess['surprise'][sess['stim_q']])
        io_entropy.append(sess['entropy'][sess['stim_q']])
        is_not_missed.append(sess['is_not_missed'])

        if len(sess['misses_Qidx']) > 0:
            print('misses trials: ', len(sess['misses_Qidx']))

    all_IO_prob[subject_id] = np.concatenate(io_prob)
    all_IO_conf[subject_id] = np.concatenate(io_conf)
    all_IO_surp[subject_id] = np.concatenate(io_surp)
    all_IO_entr[subject_id] = np.concatenate(io_entropy)
    all_not_missed_trials[subject_id] = np.concatenate(is_not_missed)

    # get subject and IO correlations
    corr_dict['r_prob'][subject_id] = np.corrcoef(
        all_IO_prob[subject_id][all_not_missed_trials[subject_id]],
        all_sub_prob[subject_id][all_not_missed_trials[subject_id]])[1, 0]
    corr_dict['r_conf'][subject_id] = np.corrcoef(
        all_IO_conf[subject_id][all_not_missed_trials[subject_id]],
        all_sub_conf[subject_id][all_not_missed_trials[subject_id]])[1, 0]

    # residuals correlation after regressing out the ideal surprise, the entropy of
    # the subject's probability and and order 2 polynome of the subject' probability
    X_confounds = np.vstack([np.concatenate(io_surp),
                             compute_entropy(all_sub_prob[subject_id]),
                             all_sub_prob[subject_id],
                             all_sub_prob[subject_id]**2]).T
    reg_confounds = LinearRegression().fit(X_confounds[all_not_missed_trials[subject_id], :],
                                           all_sub_conf[subject_id][all_not_missed_trials[subject_id], np.newaxis])
    residuals_subj_conf = np.nan*np.ones(len(all_sub_conf[subject_id]))
    residuals_subj_conf[all_not_missed_trials[subject_id]] = (
        all_sub_conf[subject_id][all_not_missed_trials[subject_id], np.newaxis]
        - reg_confounds.predict(X_confounds[all_not_missed_trials[subject_id], :])).flatten()
    all_sub_res_conf[subject_id] = residuals_subj_conf
    all_sub_entr[subject_id] = compute_entropy(all_sub_prob[subject_id])
    corr_dict['res_r_conf'][subject_id] = np.corrcoef(
        residuals_subj_conf[all_not_missed_trials[subject_id]],
        all_IO_conf[subject_id][all_not_missed_trials[subject_id]])[0, 1]

# create results dictionary
behav_io_data_all_sub = {'sub_prob': all_sub_prob,
                         'sub_conf': all_sub_conf,
                         'sub_entr': all_sub_entr,
                         'sub_res_conf': all_sub_res_conf,
                         'io_prob': all_IO_prob,
                         'io_conf': all_IO_conf,
                         'io_entr': all_IO_entr,
                         'io_surp': all_IO_surp,
                         'not_missed_trials': all_not_missed_trials}


# save data
if SAVE:
    with open(os.path.join(OUT_DIR, 'correlations_IO_all_sub.pickle'), 'wb') as f:
        pickle.dump(corr_dict, f)

    with open(os.path.join(OUT_DIR, 'behav_io_data_all_sub.pickle'), 'wb') as f:
        pickle.dump(behav_io_data_all_sub, f)