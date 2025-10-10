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
        if all(x not in col for x in ('session', 'mvt')):
            dmtx_z[col] = zscore(dmtx_z[col])
    return dmtx_z

def clean_regs(reg, tr):
    """
    Apply the same cleaning process to the given regressors as applied to the
    BOLD signal, i.e. first detrending and then high-pass filtering.
    """    
    skip_regs = ['mvt', 'stim', 'q_conf', 'q_prob', 'cue','out', 'qA_val', 'qA_conf', 'qB_val', 'qB_conf']  

    if any(keyword in reg.name for keyword in skip_regs):
        return reg.values

    return nilearn.signal.clean(reg.values.reshape(-1, 1),
                                detrend=True,
                                high_pass=params.hpf,
                                t_r=tr,
                                standardize=False)

def create_design_matrix(events_all,
                        tr,
                        frame_times,
                        subject,
                        sess):
    '''
    This function creates a design matrix from the event dataframe and adds movement regressors.
    
    Parameters
    ----------
     -event: dataframe, containinng onset, duration, tiral_type and modulation
    - tr: tr in sec
    - frame_times (corrected for slice timing correction)
    - subject
    - sess: curent session 
    Returns
    -------
    design_matrix as a pandas.DataFrame
    The rows correspond to different time frames. The columns correspond to different regressors.
    '''    

    # Get the motion regressors and events for the current session
    mvts = mf.get_mvt_reg(params.db, subject, sess)
    mvts_names = [name for name in mvts.columns]

    # make design matrix
    dmtx = make_first_level_design_matrix(frame_times=frame_times, 
                                            events=events_all, 
                                            drift_model = None,
                                            add_regs=mvts,          
                                            add_reg_names=mvts_names)
    
    # clean all regressors (exluding movement) 
    for reg in dmtx.columns:
        dmtx[reg] = clean_regs(dmtx[reg],tr)
    
    if params.zscore_per_session:
        dmtx = zscore_regressors(dmtx)

    return dmtx