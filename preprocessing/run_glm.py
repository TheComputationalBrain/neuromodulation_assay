"""
script to optain the fMRI effect m,aps used in the analysis. 

This gets all the necessary information and estimates a general linear model (GLM) that include
surprise, confidence, predictability and predictions as parametric modulation of stimuli onsets in regressors. These estimates are optained from an ideal observer model.

The output corresponds to the labels and estimates of the GLM.

This script is run for each dataset individually.
"""

import os
import sys
#specify the number of threads to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import matplotlib.pyplot as plt
from pathlib import Path
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
import utils.main_funcs as mf
import utils.fmri_funcs as fun
import utils.io_funcs as iof
from config.loader import load_config
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from TransitionProbModel.MarkovModel_Python import GenerateSequence as sg


TASK = 'EncodeProb'
SAVE_DMTX_PLOT = True
REDO_MASK = False

params, paths, _ = load_config(TASK, return_what='all')

# Init paths
beh_dir  = mf.get_beh_dir(TASK, paths)
json_file_dir = mf.get_json_dir(TASK, paths)
fmri_dir = mf.get_fmri_dir(TASK, paths)

#adjust naming
if TASK in ['NAConf']:
    if params.remove_trials:
         add_info = '_firstTrialsRemoved'
if not params.zscore_per_session:
    add_info = '_zscoreAll'
else:
     add_info = ""

#make output directories
fmri_arr_dir  = os.path.join(paths.home_dir,TASK,params.mask,'first_level',f'data_arrays_whole_brain_{params.smoothing_fwhm}') 
os.makedirs(fmri_arr_dir, exist_ok =True)

if params.update:
    output_dir = os.path.join(paths.home_dir,TASK,params.mask,'first_level', 'update_model')
    design_dir = os.path.join(output_dir, 'designmatrix_update')
else:
    output_dir = os.path.join(paths.home_dir,TASK,params.mask,'first_level') 

    design_dir = os.path.join(output_dir, 'designmatrix_nilearn')
    
os.makedirs(output_dir, exist_ok = True) 
os.makedirs(design_dir, exist_ok = True)

subjects = mf.get_subjects(TASK, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore] 
#for Explore the subjects are already removed in the fMRI data folder/in NAConf some are already removed 


for sub in subjects:
    
    print(f"--- processing subject {sub} ----")

    #get global info that's not session specific
    sessions = fun.get_sessions(sub, params)
    tr = fun.get_tr(TASK, sub, 1, json_file_dir) # in seconds
    masker = fun.get_masker(fmri_dir,tr, params.smoothing_fwhm, params, paths)

    #fMRI data
    fmri_data = os.path.join(fmri_arr_dir, f'*{sub:02d}_data*.npy')
    fmri_f = glob.glob(fmri_data)

    # check existence of the arrays
    if len(fmri_f) == 1 and REDO_MASK == False:
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
            TASK)
    
    if params.zscore_per_session == False:
        fmri_data = zscore(fmri_data)  

    design_matrix = []

    #create design matrix 
    for s,sess in enumerate(sessions):
        #IO inference 
        seq = fun.get_seq(db=TASK,
                        sub=sub,
                        sess=sess,
                        beh_dir=beh_dir)
        seq = sg.ConvertSequence(seq)['seq']

        if TASK == 'Explore':
            events = fun.get_events(TASK, sub, sess, params)
        else:
            io_inference = iof.get_post_inference(seq=seq,
                                                    seq_type= params.seq_type, 
                                                    options=params.io_options)
            events = fun.get_events(TASK, sub, sess, beh_dir, io_inference, seq) 
                  
        #frame time 
        frame_times = fun.get_fts(TASK, sub, sess, fmri_dir, json_file_dir, paths) 

        #wrapper for design matrix, uses the nilearn function within
        dmtx = create_design_matrix(events,
                                    tr, 
                                    frame_times,
                                    sub,
                                    sess,
                                    params)
        
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
            f'sub-{sub:02d}_design_matrix_{TASK}_{add_info}.pickle')) 

    # plot and save the design matrix
    if SAVE_DMTX_PLOT:
        fig_fname = f'sub-{sub:02d}_design_matrix_{TASK}_{add_info}..png'
        fig_fpath = os.path.join(design_dir, fig_fname)
        if TASK == 'Explore':
            fig, ax = plt.subplots(figsize=[8, 12])
        else:
            fig, ax = plt.subplots(figsize=[8, 6])
        plot_design_matrix(design_matrix, rescale = False, ax=ax)
        fig.suptitle(f'Regressors: Subject {sub:02d}, {TASK}', y=1.05, fontweight='bold')
        fig.savefig(fig_fpath, bbox_inches='tight', dpi=220)
        plt.close()

    #run GLM on all voxels
    print("---- Running glm ----")
    labels, estimates = run_glm(fmri_data, design_matrix.values, n_jobs = 1)

    # save results
    label_fname = f'sub-{sub:02d}_{TASK}_labels_{params.mask}{add_info}.pickle'
    with open(os.path.join(output_dir, label_fname), 'wb') as f:
        pickle.dump(labels, f)
    estimates_fname = f'sub-{sub:02d}_{TASK}_estimates_{params.mask}{add_info}.pickle'
    with open(os.path.join(output_dir, estimates_fname), 'wb') as f:
        pickle.dump(estimates, f)

    # contasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrast_matrix_neg = contrast_matrix * (-1)
    contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    contrasts_neg = dict([(column, contrast_matrix_neg[i])
                          for i, column in enumerate(design_matrix.columns)])
    
    if TASK != 'Explore':
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
                        'surprise_forced': contrasts[f'US_forced'],
                        'confidence_forced': contrasts[f'EC_{params.model}_forced'],
                        'prediction_forced': contrasts[f'ER_{params.model}_forced']}
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

        #if mask is schaefer compute the mean by region
        if (params.mask == 'schaefer') & params.parcelated:
            atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
            atlas.labels = np.insert(atlas.labels, 0, "Background")
            masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
            effects_parcel = masker.fit_transform(effect_size)
            with open(os.path.join(output_dir, f'sub-{sub:02d}_{contrast_id}_{params.mask}_{params.mask_details}_effect_size{add_info}.pickle'), 'wb') as f:
                pickle.dump(effects_parcel, f)

