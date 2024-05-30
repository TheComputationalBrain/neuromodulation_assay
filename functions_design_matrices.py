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
from nilearn.glm.first_level import make_first_level_design_matrix
from parametric_modulation_functions import compute_cleaned_pmod_regs
from scipy.stats import zscore
import main_funcs as mf
from params_and_paths import *


# Define functions

def zscore_regressors(dmtx):
    """This function applies z-scoring to regressors"""
    cols = list(dmtx.columns)
    dmtx_z = copy.deepcopy(dmtx)  # initialise the design matrix with z-scored features
    for col in cols:
        if 'session' not in col:
            dmtx_z[col] = zscore(dmtx_z[col])
    return dmtx_z

def create_design_matrix(events,
                        stim_q,
                        tr,
                        frame_times,
                        io_inference,
                        db_name,
                        subject,
                        sess,
                        add_mvt_regs=True,
                        add_question_regs=True):
    '''
    This function creates a design matrix from the IO estimates.
    
    Parameters
    ----------
     -event: dataframe, copntianing onset, duration, tiral_type and response
     -stim_q: list of question stims 
    - tr: tr in sec
    - frame_times
    - io_inference: list, containing a dict comprised of IO posterior inference
    - mvt_files: dataframe of movement regressors for all sessions
    - db: data base name 
    - add_physio_regs: bool, add physio regressors
    - add_question_regs: bool, add question regressors

    Returns
    -------
    design_matrix as a pandas.DataFrame
    The rows correspond to different time frames. The columns correspond to different regressors.
    '''    

    events_stim  = events[events['trial_type'] == 'stim']
    events_stim = events_stim.sort_values('onset') 

    # IO regressors
    reg_list = []
    io_regs = pd.DataFrame({'surprise': io_inference['surprise'],
                            'confidence': io_inference['confidence_pre'],
                            'predictions': io_inference['p1_mean_array'],
                            'predictability': io_inference['entropy']})
    
    for io_regressor in io_regs.keys():

        reg = pd.DataFrame(
            compute_cleaned_pmod_regs(np.array(events_stim['onset']), 
                                        np.array(events_stim['duration']),
                                        np.array(io_regs[io_regressor]), 
                                        frame_times,
                                        tr),
            columns=[io_regressor])
        
        reg_list.append(reg)
    
    if add_mvt_regs: 
        # Get the motion regressors and events for the current session
        mvts = mf.get_mvt_reg(db_name, subject, sess, root_dir, data_dir)
        reg_list.append(mvts)
    
    if add_question_regs:

        events_qprob = events[events['trial_type'] == 'q_prob']
        events_qconf = events[events['trial_type'] == 'q_conf']

        qprob_onsets = np.array(events_qprob['onset'])
        qprob_durations = np.array(events_qprob['duration'])
        qconf_onsets = np.array(events_qconf['onset'])
        qconf_durations = np.array(events_qconf['duration'])
        
        # add the question regressors: subject's and ideal observer's confidence and p1 estimates
        reg_sub_prob = pd.DataFrame(
            compute_cleaned_pmod_regs(qprob_onsets, qprob_durations,
                np.array(events_qprob['response']), frame_times, tr),
            columns=['sub_prob'])

        reg_sub_conf = pd.DataFrame(
            compute_cleaned_pmod_regs(qconf_onsets, qconf_durations,
                np.array(events_qconf['response']), frame_times, tr),
            columns=['sub_conf'])
        
        if stim_q[-1] >= 420: #if the last element is 420, leave it out
            stim_q = stim_q[:-1]

        # compute the probability confidence ratings of the IO
        io_prob_q = np.array([io_inference['p1_mean_array'][q]
                        for q in stim_q])
        io_conf_q = np.array([-np.log(io_inference['p1_sd_array'])[q]
                        for q in stim_q])

        reg_io_prob = pd.DataFrame(
                compute_cleaned_pmod_regs(qprob_onsets, qprob_durations,
                    io_prob_q, frame_times, tr),
                columns=['IO_prob'])

        reg_io_conf = pd.DataFrame(
                compute_cleaned_pmod_regs(qconf_onsets, qconf_durations,
                    io_conf_q, frame_times, tr),
                columns=['IO_conf'])

        #TODO: this has to be adapted for the different exp since not all of them contain both question

        #seperate regressors for EncodeProb
        qregs_prob = pd.concat(
            [reg_sub_prob, reg_io_prob], axis=1)
        
        qregs_conf = pd.concat(
            [reg_sub_conf, reg_io_conf], axis=1)

        dmtx_tmp_qp =  make_first_level_design_matrix(
            frame_times=frame_times,
            events=events_qprob.drop(columns="response"), #column ist dropped just to surpress warning
            drift_model = None,
            add_regs=qregs_prob,
            add_reg_names=[name for name in qregs_prob.columns])
        dmtx_tmp_qp = dmtx_tmp_qp.drop(columns="constant") 
        dmtx_tmp_qp.reset_index(drop=True, inplace=True)

        reg_list.append(dmtx_tmp_qp)

        dmtx_tmp_qc =  make_first_level_design_matrix(
            frame_times=frame_times,
            events=events_qconf.drop(columns="response"), #column ist dropped just to surpress warning
            drift_model = None,
            add_regs=qregs_conf,
            add_reg_names=[name for name in qregs_conf.columns])
        dmtx_tmp_qc = dmtx_tmp_qc.drop(columns="constant") 
        dmtx_tmp_qc.reset_index(drop=True, inplace=True)

        reg_list.append(dmtx_tmp_qc)

    # concat all regressors        
    all_regs = pd.concat(reg_list, axis=1)
    reg_names = [name for name in all_regs.columns]

    # make matrix
    dmtx = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_stim.drop(columns="response"), #column ist dropped just to surpress warning 
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