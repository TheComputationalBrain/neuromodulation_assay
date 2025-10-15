#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:49:30 2024

@author: Alice hodapp, Maëva L'Hôtellier
"""

import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import os.path as op
from scipy.io import loadmat
import nibabel as nib
from neuromaps import transforms
from scipy.stats import zscore
from preprocessing/initialize_subject import initialize_subject
from preprocessing/params_and_paths import Params, Paths

params = Params()
paths = Paths()

def demean(x):
    """center regressors"""
    mean = np.nanmean(x)
    return x - mean

def get_json_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):
   
    json_files_dir = {'NAConf': op.join(root_dir, data_dir, 'bids_dataset'),
                    'EncodeProb': op.join(root_dir, data_dir, 'bids_dataset'),
                    'Explore': op.join(root_dir, data_dir, 'bids/raw/'),
                    'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}
    
    return json_files_dir[db_name]


def get_fmri_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):
    
    fmri_dir = {'NAConf': op.join(root_dir, data_dir, 'derivatives'),
                'EncodeProb': op.join(root_dir, data_dir, 'derivatives'),
                'Explore': op.join(root_dir, data_dir, 'bids/derivatives/fmriprep-23.1.3_MAIN'),
                'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}

    return fmri_dir[db_name]


def get_beh_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):

    beh_dir = {'NAConf': op.join(root_dir, data_dir),
               'EncodeProb': op.join(root_dir, data_dir),
               'Explore': '/home_local/EXPLORE/github/explore/2021_Continous_2armed',
               'PNAS': op.join(root_dir, data_dir)}

    return beh_dir[db_name]

def get_subjects(db, data_dir): 
    subjects = []

    if db == 'lanA':
        subjects = list(range(1, 61))
        return subjects

    if db != 'PNAS':
        folders = os.path.join(data_dir,
                            'sub-*')
        
        for folder in glob.glob(folders):
            if os.path.isdir(folder):  # Ensure the path is a directory
                # Extract the number from the filename
                number = folder.split('sub-')[1]
                subjects.append(int(number))
    else:
        folders = os.path.join(data_dir,
                            'subj*')
        
        for folder in glob.glob(folders):
            # Extract the number from the filename
            number = folder.split('subj')[1]
            subjects.append(int(number))

    return sorted(subjects)

def get_beta_dir_and_info(task):
    if task == 'Explore':
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
    elif task == 'lanA':
        beta_dir = os.path.join(paths.home_dir, paths.fmri_dir['lanA'])
    else:
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level') 

    if task == 'NAConf':
        add_info = '_firstTrialsRemoved'
    elif not params.zscore_per_session:
        add_info = '_zscoreAll'
    else:
        add_info = ""

    return beta_dir, add_info

def load_effect_map_array(sub, task, latent_var):
    beta_dir, add_info = get_beta_dir_and_info(task)

    map = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{params.mask}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()

    return map

def load_receptor_array(on_surface=False):
    if on_surface:
        receptor_data =zscore(np.load(os.path.join(paths.receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
    else:
        receptor_data = zscore(np.load(os.path.join(paths.receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))

    return receptor_data

def load_surface_effect_maps_for_cv(subjects, task, latent_var):
    beta_dir, add_info = get_beta_dir_and_info(task)

    if task == 'lanA':
        fmri_files = []
        for subj in subjects:
            subj_id = f"{subj:03d}"  
            pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
            fmri_files.extend(glob.glob(pattern))
    else:
        fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz')))
        fmri_files = []
        for file in fmri_files_all:
            basename = os.path.basename(file)
            subj_str = basename.split('_')[0]  # 'sub-XX'
            subj_id = int(subj_str.split('-')[1])  # XX as integer
            if subj_id in subjects:
                fmri_files.append(file)
    fmri_activity = []
    for file in fmri_files:
        data_vol = nib.load(file)
        effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
        data_gii = []
        for img in effect_data:
            data_hemi = img.agg_data()
            data_hemi = np.asarray(data_hemi).T
            data_gii += [data_hemi]
        effect_array = np.hstack(data_gii)    
        fmri_activity.append(effect_array) 

    return fmri_activity




