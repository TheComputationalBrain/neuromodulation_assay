#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:05:09 2024

@author: Alice Hodapp

This gets all the necessary information and estimates a general linear model (GLM) that include
surprise, confidence, predictability and predictions as parametric modulation of stimuli onsets in regressors.
The output corresponds to the labels and estimates of the GLM.
"""

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import nibabel as nib
from nilearn.glm.first_level import run_glm
from nilearn.plotting import plot_design_matrix
from nilearn.glm.contrasts import compute_contrast
from scipy.stats import zscore
from functions_design_matrices import * #TODO: don't import all
import fmri_funcs as fun
import main_funcs as mf
import io_funcs as iof

from params_and_paths import *
# import sys
# sys.path.append('/Users/alice/postdoc/NeuroModAssay')
from TransitionProbModel.MarkovModel_Python import GenerateSequence as sg

RUN_OLS = True

# Init paths
beh_dir  = mf.get_beh_dir(DB_NAME)
json_file_dir = mf.get_json_dir(DB_NAME)
fmri_dir = mf.get_fmri_dir(DB_NAME)

#make output directories
fmri_arr_dir  = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level',f'data_arrays_whole_brain_{SMOOTHING_FWHM}')
if not os.path.exists(fmri_arr_dir):
        os.makedirs(fmri_arr_dir)
if RUN_OLS:
    output_dir = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level','OLS')
else:
    output_dir = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level')

if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
design_dir = os.path.join(output_dir, 'designmatrix')
if not os.path.exists(design_dir):
        os.makedirs(design_dir)

subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

for sub in subjects:
    
    print(f"--- processing subject {sub} ----")

    #get global info that's not session specific
    sessions = mf.get_sessions(DB_NAME,sub)
    tr = fun.get_tr(DB_NAME, sub, 1, json_file_dir) # in seconds
    masker = fun.get_masker(tr, SMOOTHING_FWHM)
    masker.fit() #TODO: restructure (maybe put in masking step)

    #fMRI data
    fmri_data = os.path.join(fmri_arr_dir, f'*{sub:02d}_data*.npy')
    fmri_f = glob.glob(fmri_data)

    # check existence of the arrays
    if len(fmri_f) == 1:
        fmri_data = np.load(glob.glob(fmri_data)[0], allow_pickle=True)
    # otherwise extract the data with masker (npy array is concatenated and saved within function)
    else:
        print("--- extraction from masker ----")
        ppssing, fmri_path = fun.get_ppssing(sub, DB_NAME)
        #TODO: comments to make clear what's happening
        nii_files, fmri_data = fun.get_fmri_data( 
            masker,
            MASK_NAME,
            sub,
            fmri_arr_dir,
            ppssing,
            fmri_path, 
            DB_NAME)
        
    fmri_data = zscore(fmri_data)

    for s,sess in enumerate(sessions):
        #TODO. list of design matrixes and then concat at end to avoid deep copy
        #IO inference 
        seq = mf.get_seq(db=DB_NAME,
                        sub=sub,
                        sess=sess,
                        beh_dir=beh_dir)
        seq = sg.ConvertSequence(seq)['seq']
        constants = mf.get_constants(data_dir=beh_dir,
                                            sub=sub,
                                            sess=sess)
        options = {'p_c': constants['pJump'], 'res': RES} 
        io_inference = iof.get_post_inference(seq=seq,
                                                seq_type='bernoulli',
                                                options=options)
        
        #get events -> this should already work on all data bases
        events = mf.get_events(DB_NAME, sub, sess, beh_dir) 
        #frame time 
        frame_times = fun.get_fts(DB_NAME, sub, sess, fmri_dir, json_file_dir) 
        #questions
        q_list = constants['StimQ'][0] 
        q_list = [int(q) for q in q_list]

        #TODO: use standard nilearn function? -> this is just a wrapper, the standard is used within
        dmtx = create_design_matrix(events,
                                    q_list,
                                    tr, 
                                    frame_times,
                                    io_inference,
                                    DB_NAME,
                                    sub,
                                    sess)
        
        # specify the session 
        dmtx['session1'], dmtx['session2'], dmtx['session3'], dmtx['session4'] = [0, 0, 0, 0]
        dmtx[f'session{s+1}'] = 1  
        dmtx = dmtx.drop(columns="constant") 
        # concatenate the sessions
        if s == 0:
            dmtx2 = dmtx #TODO: deep copy
        else:
            dmtx2 = pd.concat([dmtx2, dmtx])

        dmtx2.reset_index(drop=True, inplace=True)

    # z-score the design matrix (over all session at once, rather than session-wise)
    design_matrix = zscore_regressors(dmtx2)

    # save design matrix
    design_matrix.to_pickle(
        os.path.join(
            design_dir,
            f'sub-{sub:02d}_design_matrix_' + DB_NAME + '.pickle')) 

    # plot and save the design matrix
    if SAVE_DMTX_PLOT:
        fig_fname = f'sub-{sub:02d}_design_matrix_' + DB_NAME + '.png'
        fig_fpath = os.path.join(design_dir, fig_fname)
        fig, ax = plt.subplots(figsize=[8, 6])
        plot_design_matrix(design_matrix, rescale = False, ax=ax)
        fig.suptitle(f'Regressors: Subject {sub:02d}, {DB_NAME}', y=1.05, fontweight='bold')
        fig.savefig(fig_fpath, bbox_inches='tight', dpi=220)

    # run GLM on all voxels
    if RUN_OLS:    
        print("---- Running glm with OLS ----")
        labels, estimates = run_glm(fmri_data, design_matrix.values, noise_model='ols', n_jobs = 10)
    else: 
        print("---- Running glm with autoregressive model  ----")
        labels, estimates = run_glm(fmri_data, design_matrix.values, n_jobs = 10)

    # save results
    label_fname = f'sub-{sub:02d}_{DB_NAME}_labels_{MASK_NAME}.pickle'
    with open(os.path.join(output_dir, label_fname), 'wb') as f:
        pickle.dump(labels, f)
    estimates_fname = f'sub-{sub:02d}_{DB_NAME}_estimates_{MASK_NAME}.pickle'
    with open(os.path.join(output_dir, estimates_fname), 'wb') as f:
        pickle.dump(estimates, f)

    # contasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    contrasts = {'surprise': contrasts['surprise'],
                    'confidence': contrasts['confidence'],
                    'predictions': contrasts['predictions'],
                    'predictability': contrasts['predictability']}
    
    for contrast_id in contrasts:
        contrast = compute_contrast(labels,
                                    estimates,
                                    contrasts[contrast_id],
                                    stat_type ='t')

        # Save t-map
        t_val = masker.inverse_transform(contrast.stat())
        fname = f'sub-{sub:02d}_{contrast_id}_{MASK_NAME}_tmap.nii.gz'
        nib.save(t_val, os.path.join(output_dir, fname))

        #save effect size = beta 
        #save in a pickle format
        with open(os.path.join(output_dir, 
                                    f'sub-{sub:02d}_{contrast_id}_{MASK_NAME}_effect_size_map.pickle'), 'wb') as f:
                    pickle.dump(contrast.effect_size(), f)
        #save in nii format
        effect_size = masker.inverse_transform(contrast.effect_size())
        fname = f'sub-{sub:02d}_{contrast_id}_{MASK_NAME}_effect_size_map.nii.gz'
        nib.save(effect_size, os.path.join(output_dir, fname))

        #TODO: work with nifti file also for glm data to make it less error prone

