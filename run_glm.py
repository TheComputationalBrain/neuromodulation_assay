#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:05:09 2024

@author: Alice Hodapp

This gets all the necessary information and estimates a general linear model (GLM) that include
surprise, confidence, predictability and predictions as parametric modulation of stimuli onsets in regressors.
The output corresponds to the labels and estimates of the GLM.
"""

import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
from nilearn.glm.first_level import run_glm
from nilearn.plotting import plot_design_matrix
from nilearn.glm.contrasts import compute_contrast
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.stats import zscore
from functions_design_matrices import create_design_matrix, zscore_regressors
import fmri_funcs as fun
import main_funcs as mf
import io_funcs as iof

from params_and_paths import Params, Paths
from TransitionProbModel.MarkovModel_Python import GenerateSequence as sg

SAVE_DMTX_PLOT = True

paths = Paths()
params = Params()

# Init paths
beh_dir  = mf.get_beh_dir(params.db)
json_file_dir = mf.get_json_dir(params.db)
fmri_dir = mf.get_fmri_dir(params.db)

#adjust naming
if params.db in ['NAConf']:
    if params.remove_trials:
         add_info = '_firstTrialsRemoved'
if not params.zscore_per_session:
    add_info = '_zscoreAll'
else:
     add_info = ""

#make output directories
fmri_arr_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level',f'data_arrays_whole_brain_{params.smoothing_fwhm}') 
if not os.path.exists(fmri_arr_dir):
        os.makedirs(fmri_arr_dir)

if params.update:
    if params.db == 'Explore':
        output_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model', params.model)
    else:
        output_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model')
    design_dir = os.path.join(output_dir, 'designmatrix_update')
else:
    if params.db == 'Explore':
        output_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level',params.model)
    else:
        output_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level') 

    design_dir = os.path.join(output_dir, 'designmatrix_nilearn')
    
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
if not os.path.exists(design_dir):
        os.makedirs(design_dir)

subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore] 
#for Explore the subjects are already removed in the fMRI data folder/in NAConf some are already removed 


for sub in subjects:
    
    print(f"--- processing subject {sub} ----")

    #get global info that's not session specific
    sessions = mf.get_sessions(sub)
    tr = fun.get_tr(params.db, sub, 1, json_file_dir) # in seconds
    masker = fun.get_masker(fmri_dir,tr, params.smoothing_fwhm)

    #fMRI data
    fmri_data = os.path.join(fmri_arr_dir, f'*{sub:02d}_data*.npy')
    fmri_f = glob.glob(fmri_data)

    # check existence of the arrays
    if len(fmri_f) == 1 and params.redo_mask == False:
        fmri_data = np.load(glob.glob(fmri_data)[0], allow_pickle=True)
    # otherwise extract the data with masker 
    else:
        print("--- extraction from masker ----")
        #mask the correct nii files for study and saves + returns all runs as one numpy array
        nii_files, fmri_data = fun.get_fmri_data( 
            masker,
            params.mask,
            sub,
            fmri_arr_dir, 
            params.db)
    
    if params.zscore_per_session == False:
        fmri_data = zscore(fmri_data)  

    design_matrix = []

    #create design matrix 
    for s,sess in enumerate(sessions):
        #IO inference 
        seq = mf.get_seq(db=params.db,
                        sub=sub,
                        sess=sess,
                        beh_dir=beh_dir)
        seq = sg.ConvertSequence(seq)['seq']

        if params.db == 'Explore':
            events = mf.get_events(params.db, sub, sess)
        else:
            io_inference = iof.get_post_inference(seq=seq,
                                                    seq_type= params.seq_type, 
                                                    options=params.io_options)
            events = mf.get_events(params.db, sub, sess, beh_dir, io_inference, seq) 
                  
        #frame time 
        frame_times = fun.get_fts(params.db, sub, sess, fmri_dir, json_file_dir) 

        #wrapper for design matrix, uses the nilearn function within
        dmtx = create_design_matrix(events,
                                    tr, 
                                    frame_times,
                                    sub,
                                    sess)
        
        # specify the session 
        for i in range(1, len(sessions) + 1):
            dmtx[f'session{i}'] = 0
        dmtx[f'session{s+1}'] = 1
        dmtx = dmtx.drop(columns="constant")

        design_matrix.append(dmtx)

    # concatenate the sessions
    design_matrix = pd.concat(design_matrix)

    # z-score the design matrix (over all session at once, rather than session-wise)
    if params.zscore_per_session == False:
        design_matrix = zscore_regressors(design_matrix)

    # save design matrix
    design_matrix.to_pickle(
        os.path.join(
            design_dir,
            f'sub-{sub:02d}_design_matrix_' + params.db + f'{add_info}.pickle')) 

    # plot and save the design matrix
    if SAVE_DMTX_PLOT:
        fig_fname = f'sub-{sub:02d}_design_matrix_' + params.db + f'{add_info}.png'
        fig_fpath = os.path.join(design_dir, fig_fname)
        if params.db == 'Explore':
            fig, ax = plt.subplots(figsize=[8, 12])
        else:
            fig, ax = plt.subplots(figsize=[8, 6])
        plot_design_matrix(design_matrix, rescale = False, ax=ax)
        fig.suptitle(f'Regressors: Subject {sub:02d}, {params.db}', y=1.05, fontweight='bold')
        fig.savefig(fig_fpath, bbox_inches='tight', dpi=220)
        plt.close()

    #run GLM on all voxels
    print("---- Running glm with autoregressive model  ----")
    labels, estimates = run_glm(fmri_data, design_matrix.values, n_jobs = 1)

    # save results
    label_fname = f'sub-{sub:02d}_{params.db}_labels_{params.mask}{add_info}.pickle'
    with open(os.path.join(output_dir, label_fname), 'wb') as f:
        pickle.dump(labels, f)
    estimates_fname = f'sub-{sub:02d}_{params.db}_estimates_{params.mask}{add_info}.pickle'
    with open(os.path.join(output_dir, estimates_fname), 'wb') as f:
        pickle.dump(estimates, f)

    # contasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrast_matrix_neg = contrast_matrix * (-1)
    contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    contrasts_neg = dict([(column, contrast_matrix_neg[i])
                          for i, column in enumerate(design_matrix.columns)])
    
    if params.db != 'Explore':
        if params.update:
            contrasts = {'update': contrasts['update'],
                        'predictions': contrasts['predictions'],
                        'predictability': contrasts['predictability']}
        else:
            contrasts = {'surprise': contrasts['surprise'],
                        'confidence': contrasts['confidence'],
                        'predictions': contrasts['predictions'],
                        'predictability': contrasts['predictability'],
                        'surprise_neg': contrasts_neg['surprise'],
                        'confidence_neg': contrasts_neg['confidence'],
                        'predictions_neg': contrasts_neg['predictions'],
                        'predictability_neg': contrasts_neg['predictability']
                        }
    else:
        if params.split: 
            contrasts = {'surprise_free': contrasts['US_free'],
                        'confidence_free': contrasts[f'EC_{params.model}_free'],
                        'prediction_free': contrasts[f'ER_{params.model}_free'],
                        'predictability_free': contrasts[f'entropy_{params.model}_free'],
                        'surprise_forced': contrasts[f'US_forced'],
                        'confidence_forced': contrasts[f'EC_{params.model}_forced'],
                        'prediction_forced': contrasts[f'ER_{params.model}_forced'],
                        'predictability_forced': contrasts[f'entropy_{params.model}_forced']}
        elif params.model == 'US_reward':
            contrasts = {'surprise': contrasts['US'],
                        'confidence': contrasts[f'EC_chosen'],
                        'predictions': contrasts[f'ER_chosen'],
                        'predictability': contrasts[f'entropy_chosen'],
                        'surprise_neg': contrasts_neg['US'],
                        'confidence_neg': contrasts_neg[f'EC_chosen'],
                        'predictions_neg': contrasts_neg[f'ER_chosen'],
                        'predictability_neg': contrasts_neg[f'entropy_chosen']}
        elif params.model in ['noEntropy', 'noEntropy_reducedDM', 'noEntropy_reducedDM2']:
            contrasts = {'surprise': contrasts['US'],
                        'confidence': contrasts[f'EC_chosen'],
                        'predictions': contrasts[f'ER_chosen'],
                        'surprise_neg': contrasts_neg['US'],
                        'confidence_neg': contrasts_neg[f'EC_chosen'],
                        'predictions_neg': contrasts_neg[f'ER_chosen']}
        elif params.model in ['noEntropy_noER']:
            contrasts = {'surprise': contrasts['US'],
                        'confidence': contrasts[f'EC_chosen'],
                        'surprise_neg': contrasts_neg['US'],
                        'confidence_neg': contrasts_neg[f'EC_chosen']}

            
    for contrast_id in contrasts:
        contrast = compute_contrast(labels,
                                    estimates,
                                    contrasts[contrast_id],
                                    stat_type ='t')

        # Save z-map
        z_val = masker.inverse_transform(contrast.z_score())
        fname = f'sub-{sub:02d}_{contrast_id}_{params.mask}_zmap{add_info}.nii.gz'
        nib.save(z_val, os.path.join(output_dir, fname))

        #save effect size = beta 
        #save in a pickle format
        with open(os.path.join(output_dir, 
                                    f'sub-{sub:02d}_{contrast_id}_{params.mask}_effect_size{add_info}.pickle'), 'wb') as f:
                    pickle.dump(contrast.effect_size(), f)
        #save in nii format
        effect_size = masker.inverse_transform(contrast.effect_size())
        fname = f'sub-{sub:02d}_{contrast_id}_{params.mask}_effect_size_map{add_info}.nii.gz'
        nib.save(effect_size, os.path.join(output_dir, fname))

        #if mask is schaefer compute the mean by region, as we only use this atlas for the autoradiography data that's only available in the Schaefer 100 parcelation
        if (params.mask == 'schaefer') & params.parcelated:
            atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
            atlas.labels = np.insert(atlas.labels, 0, "Background")
            masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
            effects_parcel = masker.fit_transform(effect_size)
            with open(os.path.join(output_dir, f'sub-{sub:02d}_{contrast_id}_{params.mask}_{params.mask_details}_effect_size{add_info}.pickle'), 'wb') as f:
                pickle.dump(effects_parcel, f)
        elif (params.mask == 'desikan') & params.parcelated:
            atlas = fetch_desikan_killiany() 
            masker = NiftiLabelsMasker(labels_img=atlas['image']) #parcelate
            effects_parcel = masker.fit_transform(effect_size)
            with open(os.path.join(output_dir, f'sub-{sub:02d}_{contrast_id}_{params.mask}_{params.mask_details}_effect_size{add_info}.pickle'), 'wb') as f:
                pickle.dump(effects_parcel, f)
