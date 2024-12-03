#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alice Hodapp 

This script computes group level estimates based on first-level
contrasts and plots the results. First-level contrasts have been parformed seperatly. 

"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting, image
from nilearn import datasets
from nilearn import surface
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from sklearn.linear_model import LinearRegression
import main_funcs as mf
import fmri_funcs as fun
import nibabel as nib
from scipy.stats import zscore
from params_and_paths import Paths, Params, Receptors
from dominance_stats import dominance_stats
from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_atlas_schaefer_2018
from multiprocessing import Pool


PLOT_ONLY = True
NUM_WORKERS = 1 #set number of CPUs in case dominance analysis is run on group level data 

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level')
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if params.update:
    if params.db == 'Explore':
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model', params.model)
        plot_path = os.path.join(output_dir, params.model, 'plot_raw_update')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
    else:
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model')
        plot_path = os.path.join(output_dir, 'plot_raw_update')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
else:
    if params.db == 'Explore':
        plot_path = os.path.join(output_dir, params.model, 'plot_raw')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level',params.model)
    else:
        plot_path = os.path.join(output_dir, 'plot_raw')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level')

for var in params.latent_vars:
    print(f'------- running group analysis for {var} -------')

    eff_size_files = []
    eff_size_arrays = []

    for sub in subjects:
        #get contrast data (beta estimates) and concatinate
        ef_size = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{var}_{params.mask}_effect_size_map.nii.gz'))
        eff_size_files.append(ef_size)
        # eff_size_arrays.append(masker.fit_transform(ef_size).flatten())

    #second level model:
    design_matrix = pd.DataFrame([1] * len(eff_size_files),
                                 columns=['intercept']) #one sample test
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(eff_size_files,
                                                design_matrix=design_matrix)

    #mean beta
    plt.rcParams.update({'font.size': 8})
    effect_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='effect_size')
    fname = f'{var}_{params.mask}_effect_map.nii.gz'
    nib.save(effect_map, os.path.join(output_dir, fname))
    plotting.plot_img_on_surf(effect_map, surf_mesh='fsaverage5', 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True, cbar_tick_format='%.2f')
    fname = f'{var}_effect_map.png' 
    plt.savefig(os.path.join(plot_path, fname))

    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(effect_map, fsaverage.pial_right)
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        colorbar=True,
        bg_map=fsaverage.sulc_right)
    fname = f'{var}_effect_map_right.png' 
    plt.savefig(os.path.join(plot_path, fname))

    #z map 
    z_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='z_score')

    #plot both hemis and lateral + medial
    fname = f'{var}_{params.mask}_effect_map.nii.gz'
    nib.save(z_map, os.path.join(output_dir, fname))
    plotting.plot_img_on_surf(z_map, surf_mesh='fsaverage5', 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True, cbar_tick_format='%.2f')
    fname = f'{var}_z_map.png' 
    plt.savefig(os.path.join(plot_path, fname))

    #sanity check
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(z_map, fsaverage.pial_right)
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        colorbar=True,
        bg_map=fsaverage.sulc_right)
    fname = f'{var}_z_map_right.png' 
    plt.savefig(os.path.join(plot_path, fname))

