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

def demean(x):
    """center regressors"""
    return x - np.mean(x)

def clean_regs(regs, tr):
    """
    Apply the same cleaning process to the given regressors as applied to the
    BOLD signal, i.e. first detrending and then high-pass filtering.
    """    
    return nilearn.signal.clean(regs,
        detrend=True,
        high_pass=params.hpf, t_r=tr,
        standardize=False
   )

def get_IO_question_estimates(events):
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

    events_stim  = events[events['trial_type'] == 'stim'].copy()
    events_stim = events_stim.sort_values('onset') 

    # IO regressors
    reg_list = []
    io_regs = pd.DataFrame({'surprise': io_inference['surprise'],
                            'confidence': io_inference['confidence_pre'],
                            'predictions': io_inference['p1_mean_array'],
                            'predictability': io_inference['entropy']})

    for io_regressor in io_regs.keys():

        events_stim['modulation'] = demean(io_regs[io_regressor])
        reg = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_stim, 
                                            drift_model = None)   
        reg.drop(columns=['constant'], inplace=True) #we will add a stim regressor in the final design matrix 
        reg.rename(columns={'stim': io_regressor}, inplace=True)
        reg[io_regressor] = clean_regs(reg[io_regressor].values.reshape(-1,1), tr)
        reg.reset_index(drop=True, inplace=True)

        reg_list.append(reg)

    # Get the motion regressors and events for the current session
    mvts = mf.get_mvt_reg(params.db, subject, sess)
    reg_list.append(mvts)

    ### question regressors 

    # add the question regressors: subject's and ideal observer's confidence and p1 estimates

    if params.db == 'EncodeProb':
        events_qprob = events[events['trial_type'] == 'q_prob'].copy() #events contains the subject responses as modulation
        events_qprob['modulation'] = demean(events_qprob['modulation'])
        reg_sub_prob = make_first_level_design_matrix(frame_times=frame_times, 
                                                events=events_qprob, 
                                                drift_model = None)   
        reg_sub_prob.drop(columns=['constant'], inplace=True)
        reg_sub_prob.rename(columns={'q_prob': 'sub_prob'}, inplace=True)
        reg_sub_prob['sub_prob'] = clean_regs(reg_sub_prob['sub_prob'].values.reshape(-1,1), tr)
        reg_sub_prob.reset_index(drop=True, inplace=True)

        reg_list.append(reg_sub_prob)

    events_qconf = events[events['trial_type'] == 'q_conf'].copy()
    events_qconf['modulation'] = demean(events_qconf['modulation'])
    reg_sub_conf = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_qconf, 
                                            drift_model = None)  
    reg_sub_conf.drop(columns=['constant'], inplace=True)
    reg_sub_conf.rename(columns={'q_conf': 'sub_conf'}, inplace=True)
    reg_sub_conf['sub_conf'] = clean_regs(reg_sub_conf['sub_conf'].values.reshape(-1,1), tr)
    reg_sub_conf.reset_index(drop=True, inplace=True)

    reg_list.append(reg_sub_conf)

    # compute the probability confidence ratings of the IO
    stim_q = get_IO_question_estimates(events)

    stim_count = events[events['trial_type'] == 'stim'].shape[0]
    if stim_q[-1] >= stim_count: 
        stim_q = stim_q[:-1]

    if params.db == 'EncodeProb':
        io_prob_q = np.array([io_inference['p1_mean_array'][q]
                        for q in stim_q])
        events_qprob['modulation'] = demean(io_prob_q)
        reg_io_prob = make_first_level_design_matrix(frame_times=frame_times, 
                                                events=events_qprob, 
                                                drift_model = None)   
        reg_io_prob.drop(columns=['constant'], inplace=True) 
        reg_io_prob.rename(columns={'q_prob': 'IO_prob'}, inplace=True)
        reg_io_prob['IO_prob'] = clean_regs(reg_io_prob['IO_prob'].values.reshape(-1,1), tr)
        reg_io_prob.reset_index(drop=True, inplace=True)

        reg_list.append(reg_io_prob)

    io_conf_q = np.array([-np.log(io_inference['p1_sd_array'])[q]
                for q in stim_q])
    events_qconf['modulation'] = demean(io_conf_q)
    reg_io_conf = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_qconf, 
                                            drift_model = None)   
    reg_io_conf.drop(columns=['constant'], inplace=True) 
    reg_io_conf.rename(columns={'q_conf': 'IO_conf'}, inplace=True)
    reg_io_conf['IO_conf'] = clean_regs(reg_io_conf['IO_conf'].values.reshape(-1,1), tr)
    reg_io_conf.reset_index(drop=True, inplace=True)

    reg_list.append(reg_io_conf)

    # concat all regressors        
    all_regs = pd.concat(reg_list, axis=1)
    reg_names = [name for name in all_regs.columns]

    # make matrix
    dmtx = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events.drop(columns="modulation"), #set all event regressors to 1
                                            drift_model = None,
                                            add_regs=all_regs,          
                                            add_reg_names=reg_names)
    
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