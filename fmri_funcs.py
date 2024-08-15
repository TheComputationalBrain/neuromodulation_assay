#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:49:30 2024

@author: Alice Hodapp, Maëva L'Hôtellier
"""

import glob
import numpy as np
import os.path as op
import nibabel as nib
import abagen
from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_atlas_schaefer_2018
from nilearn import image
from nilearn.input_data import MultiNiftiMasker 
from main_funcs import *
from params_and_paths import Params, Paths

params = Params()
paths = Paths()

def get_masker(tr, smoothing_fwhm):
    
    #mask_path = os.path.join(mask_dir[DATA_ACCESS], MASK)

    if params.mask == 'harvard_oxford_cortical':
            atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    elif params.mask == 'harvard_oxford_subcortical':
        atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    elif params.mask == 'schaefer':
        atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2)
        atlas.labels = np.insert(atlas.labels, 0, "Background")
    elif params.mask == 'desikan':
        atlas = abagen.fetch_desikan_killiany() 
    else:
        raise ValueError("Unknown atlas!")
    
    if params.mask != 'desikan':
        atlas_img = image.load_img(atlas.maps)
        mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #mask background
    else:
        atlas_img = image.load_img(atlas['image'])
        mask = ~np.isin(image.get_data(atlas_img), [0,35,36,37,38,39,40,41,76,77,78,79,80,81,82,83])
        mask_img = image.new_img_like(atlas_img, mask) #only cortical structures 

    masker = MultiNiftiMasker(
        mask_img=mask_img,
        detrend=True,  # Improves the SNR by removing linear trends
        high_pass=params.hpf,  # kept small to keep sensitivity,
        standardize=False,
        smoothing_fwhm=smoothing_fwhm,
        t_r=tr,
    )
    
    return masker

def get_tr(db, sub, sess, data_dir):
    """This function fetches and returns the repetition time featured in an
    experimental run in the MRI scanner for a given subject and run."""
    if db == 'PNAS':
        infofilepath = op.join(data_dir,
                                    f'subj{sub:02d}',
                                    'preprocEPI', 'SliceTimingInfo.mat') 
        data = loadmat(infofilepath)
        tr = int(data['TR'])

    else:
        json_file = import_json_info(data_dir, sub, sess)
        tr = json_file['RepetitionTime']

    return tr

def fetch_n_scans(data):
    """
    This function compute the number of scans from the trigger timings by
    subtracting the last saved time by the first one.
    Inputs
    ------
    data: str, specification of the file
    Returns
    -------
    n_scans: int, number of scans
    """
    return nib.load(data).header['dim'][4]

def demean(x):
    return x - np.nanmean(x)


def get_fts(db, sub, sess, fmri_dir, json_dir):
        
    if db == 'PNAS':
        epi_dir =  op.join(fmri_dir, f'subj{sub:02d}','preprocEPI')
        ppcessed_files = 'swtuaepi*.nii'
    
    else:
        epi_dir =  op.join(fmri_dir, f'sub-{sub:02d}')
        ppcessed_files = 'wtrasub*.nii'
        if db == 'EncodeProb':
            if sub not in [2, 16]:
                ppcessed_files = 'wrtrasub*.nii'
        
    if db == 'Explore':
        ppcessed_files = '*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        epi_dir =  op.join(fmri_dir, f'sub-{sub:02d}/func')

    fmri_files = sorted(glob.glob(os.path.join(epi_dir, ppcessed_files)))

    if db == 'PNAS':
        sessfiles = [f for f in fmri_files if f"sess{sess}" in f]
    else:
        sessfiles = [f for f in fmri_files if f"run-0{sess}" in f]

    n_scans = fetch_n_scans(sessfiles[0]) 
    tr = get_tr(db, sub, sess, json_dir)
    frame_times = (np.arange(n_scans)*tr).flatten() #?n starts with 0 because of slice time correction to first slide- could be tr instead: np.cumsum([tr]* n_scans)

    if db == 'EncodeProb':
        frame_times = frame_times + tr/2 #slice timing correction was to middle slice 

    return frame_times

def get_ppssing(sub, db_name):
    if db_name == 'EncodeProb':
        if sub in [2, 16]:
            ppssing = 'wtrasub'
    
        else:
            ppssing = 'wrtrasub'
    
    elif db_name == 'NAConf':
        ppssing = 'wtrasub'

    elif db_name == 'PNAS':
        ppssing = 'wtraepi'

    fmri_path = op.join(paths.root_dir, paths.data_dir,
                        f'derivatives/sub-{sub:02d}')
    
    if db_name == 'Explore':
        ppssing = 'space-MNI152NLin2009cAsym_desc-preproc_bold'
        fmri_path = get_fmri_dir(db_name)
        fmri_path = op.join(fmri_path, f'sub-{sub:02d}', 'func')
    
    if db_name == 'PNAS':
        fmri_path = op.join(paths.root_dir, paths.data_dir,
                            f'MRI_data/raw_data/subj{sub:02d}/fMRI')

    return ppssing, fmri_path

def get_nii_files(ppssing, fmri_path, db_name):
    
    if db_name != 'Explore':
        return sorted(glob.glob(op.join(fmri_path, f'{ppssing}*.nii')))
    else:
        return sorted(glob.glob(op.join(fmri_path, f'*{ppssing}.nii.gz'))) 
    
def get_fmri_data(masker, masker_id, sub, output_dir,
                  ppssing, fmri_path, db_name):

    nii_files = get_nii_files(ppssing, fmri_path, db_name)
    data_array = masker.fit_transform(nii_files) 

    fname = f'sub-{sub:02d}_data_{ppssing}_mask-{masker_id}.npy'
    
    nib.save(masker.mask_img_, op.join(output_dir,
                                       f'{masker_id}_mask_img_sub-{sub:02d}'))
    
    fMRI_data = np.vstack(data_array) # concatenate
    np.save(op.join(output_dir, fname), fMRI_data) 

    return nii_files, fMRI_data


