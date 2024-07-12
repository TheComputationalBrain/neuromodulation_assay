#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:05:09 2024

@author: Alice Hodapp

This script contrains functions to create design matrices for the probability
learning tasks
"""

# Import useful libraries
import copy
import numpy as np
import pandas as pd
import nilearn.signal
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy.stats import zscore
from params_and_paths import Params
import main_funcs as mf

params = Params()

# Define functions

def zscore_regressors(dmtx):
    """Apply z-scoring to regressors"""
    cols = list(dmtx.columns)
    dmtx_z = copy.deepcopy(dmtx)  # initialise the design matrix with z-scored features
    for col in cols:
        if 'session' not in col:
            dmtx_z[col] = zscore(dmtx_z[col])
    return dmtx_z

def clean_regs(reg, tr):
    """
    Apply the same cleaning process to the given regressors as applied to the
    BOLD signal, i.e. first detrending and then high-pass filtering.
    """    
    if not reg.name.startswith('mvt'):
        return nilearn.signal.clean(reg.values.reshape(-1,1),
            detrend=True,
            high_pass=params.hpf, t_r=tr,
            zscore_sample=False)
    return reg.values

def get_stimq(events):
    events_sorted = events.sort_values(by='onset').reset_index(drop=True)
    stim_count = 0
    stim_before_q_prob = []
    for indx, row in events_sorted.iterrows():
        if row['trial_type'] == 'stim':
            stim_count += 1
            last_stim_n = stim_count
        elif row['trial_type'] == 'q_conf':
            stim_before_q_prob.append(last_stim_n-1)
    return stim_before_q_prob

def add_io_regs(io_regs, events):
    new_rows = []

    for key, values in io_regs.items():
        stim_rows = events[events['trial_type'] == 'stim'].copy()
        if len(values) != len(stim_rows):
            raise ValueError(f"Length of stims for {key} does not match number of 'stim' rows.")
        
        values = mf.demean(values)

        for i, value in enumerate(values):
            new_row = stim_rows.iloc[i].copy()
            new_row['modulation'] = value
            new_row['trial_type'] = key
            new_rows.append(new_row)

    new_events = pd.concat([events, pd.DataFrame(new_rows)], ignore_index=True)
    return new_events

def create_design_matrix(events,
                        tr,
                        frame_times,
                        io_inference,
                        subject,
                        sess):
    '''
    This function creates a design matrix from the IO estimates.
    
    Parameters
    ----------
     -event: dataframe, copntianing onset, duration, tiral_type and response
     -stim_q: list of question stims 
    - tr: tr in sec
    - frame_times
    - io_inference: list, containing a dict comprised of IO posterior inference
    - db: data base name 
    - subject
    - sess: curent session 
    - add_physio_regs: bool, add physio regressors
    - add_question_regs: bool, add question regressors

    Returns
    -------
    design_matrix as a pandas.DataFrame
    The rows correspond to different time frames. The columns correspond to different regressors.
    '''    
    stim_q = get_stimq(events)

    stim_count = events[events['trial_type'] == 'stim'].shape[0]
    if stim_q[-1] >= stim_count: 
        stim_q = stim_q[:-1]

    # IO regressors
    io_regs = pd.DataFrame({'surprise': io_inference['surprise'],
                            'confidence': io_inference['confidence_pre'],
                            'predictions': io_inference['p1_mean_array'],
                            'predictability': io_inference['entropy']})

    events_all = add_io_regs(io_regs, events)

    # Get the motion regressors and events for the current session
    mvts = mf.get_mvt_reg(params.db, subject, sess)

    # add the question regressors for ideal observer's confidence and p1 estimates

    # compute the probability confidence ratings of the IO

    if params.db == 'EncodeProb':
        io_prob_q = np.array([io_inference['p1_mean_array'][q]
                        for q in stim_q])
        events_all.loc[events_all['trial_type'] == 'io_prob', 'modulation'] = mf.demean(io_prob_q)


    io_conf_q = np.array([-np.log(io_inference['p1_sd_array'])[q]
                for q in stim_q])
    events_all.loc[events_all['trial_type'] == 'io_conf', 'modulation'] = mf.demean(io_conf_q)

    # make matrix
    mvts_names = [name for name in mvts.columns]
    dmtx = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_all, 
                                            drift_model = None,
                                            add_regs=mvts,          
                                            add_reg_names=mvts_names)
    
    # clean regressors (exluding movement) 
    for reg in dmtx.columns:
        dmtx[reg] = clean_regs(dmtx[reg],tr)
    
    #TODO: add special case explore -> outcome regressor
    # if db =='Explore':
        
    #     # Initialize subject-specific parameters and find observed outcome values and onsets
    #     para = initialize_subject(sub)
    #     out_mag = para[para['block'] == sess + 8]['reward']
    #     out_onset = para[para['block'] == sess + 8]['outcome_start']

    #     # Create a dataframe for outcome-related regressors
    #     df_out = pd.DataFrame({'onset': list(out_onset),
    #                             'trial_type': ['out magnitude'] * len(out_onset),
    #                             'duration': [0] * len(out_onset),
    #                             'modulation': out_mag})

    #     # Create a design matrix for outcome-related regressors
    #     out_dmtx = make_first_level_design_matrix(frame_times,
    #                                         df_out,
    #                                         hrf_model='spm',
    #                                         drift_model=None)

    #     # Add the convolved outcome value regressor to the base design matrix
    #     dmtx = pd.concat((dmtx, out_dmtx['out_magnitude']), axis=1)

    #     if not 'missed' in dmtx.columns:
    #         dmtx.insert(1, 'missed', [-0.087771710942605] *  len(dmtx)) 

    return dmtx