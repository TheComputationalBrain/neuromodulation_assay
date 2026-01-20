#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alice Hodapp 

This script computes group level estimates based on first-level
contrasts and plots the results. First-level contrasts have been parformed seperatly. 

"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn import plotting
from nilearn import datasets
import utils.fmri_funcs as fun
import nibabel as nib
import utils.main_funcs as mf
from config.loader import load_config


TASK = 'Explore'

RUN_PERMUTATION = False
RESOLUTION = 'fsaverage'

params, paths, rec = load_config(TASK, return_what='all')


#for cluster permutation:
N_PERM = 100000
N_JOBS = 4
TRESH = 0.001
FWHM = 5
TWO_SIDED = False

# --- Utility functions ---

def plot_beta_map(effect_map, var, plot_path, resolution, add_info):
    """Plot and save beta (effect size) maps on surface."""
    plotting.plot_img_on_surf(
        effect_map, surf_mesh=resolution,
        hemispheres=['left', 'right'], views=['lateral', 'medial'],
        threshold=1e-50, title=f'{var} (beta map)',
        colorbar=True, cmap='cold_hot',
        inflate=True, symmetric_cbar=True,
        cbar_tick_format='%2f'
    )
    for ax in plt.gcf().axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fname = f'{var}_effect_map{add_info}.pdf'
    plt.savefig(os.path.join(plot_path, fname), transparent=True)
    plt.close()


def plot_z_map(z_map, var, plot_path, resolution, add_info):
    """Plot and save z-score maps on surface."""
    # Left + Right, both views
    plotting.plot_img_on_surf(
        z_map, surf_mesh=resolution,
        hemispheres=['left', 'right'], views=['lateral', 'medial'],
        threshold=1e-50, title=f'{var} (z-map)',
        colorbar=True, cmap='cold_hot',
        inflate=True, symmetric_cbar=True,
        cbar_tick_format='%.2f'
    )
    for ax in plt.gcf().axes:
        for coll in ax.collections:
            coll.set_rasterized(True)

    plt.savefig(os.path.join(plot_path, f'{var}_z_map{add_info}.pdf'), transparent=True)
    plt.close()

    # Right hemisphere only, lateral view
    plotting.plot_img_on_surf(
        z_map, surf_mesh=resolution,
        hemispheres=['right'], views=['lateral'],
        threshold=1e-50, colorbar=True,
        cmap='cold_hot', inflate=True,
        symmetric_cbar=True, cbar_tick_format='%i'
    )
    for ax in plt.gcf().axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    plt.savefig(os.path.join(plot_path, f'{var}_z_map_right{add_info}.pdf'), transparent=True)
    plt.close()


# --- Main analysis ---
mf.set_publication_style(font_size=7, layout="3-across")


fmri_dir = mf.get_fmri_dir(params.db, paths)
subjects = [s for s in mf.get_subjects(params.db, fmri_dir) if s not in params.ignore]

# Output paths
if params.db == 'Explore':
    output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level',params.model)
else: 
    output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level')
beta_dir, add_info  = mf.get_beta_dir_and_info(TASK, params, paths)

plot_path = os.path.join(output_dir, 'plot_raw')
os.makedirs(plot_path, exist_ok=True)

# Masker setup
masker = fun.get_masker(params=params, paths=paths)
masker.fit()

# Templates
fsaverage = datasets.fetch_surf_fsaverage(mesh=RESOLUTION)

# --- Run second-level analysis ---
for var in params.latent_vars:
    print(f'------- running group analysis for {var} -------')

    eff_size_files = [
        nib.load(os.path.join(beta_dir, f'sub-{sub:02d}_{var}_{params.mask}_effect_size_map{add_info}.nii.gz'))
        for sub in subjects
    ]

    # Second-level model
    design_matrix = pd.DataFrame({'intercept': [1] * len(eff_size_files)})
    second_level_model = SecondLevelModel().fit(eff_size_files, design_matrix=design_matrix)

    # --- Effect (beta) map ---
    effect_map = second_level_model.compute_contrast('intercept', output_type='effect_size')
    nib.save(effect_map, os.path.join(output_dir, f'{var}_{params.mask}_effect_map{add_info}.nii.gz'))

    # Save numpy version
    effect = masker.transform(effect_map)
    with open(os.path.join(output_dir, f'group_{var}_{params.mask}_effect_size{add_info}.pickle'), 'wb') as f:
        pickle.dump(effect, f)

    # plot_beta_map(effect_map, var, plot_path, RESOLUTION, add_info)

    # --- Z-map ---
    z_map = second_level_model.compute_contrast('intercept', output_type='z_score')
    nib.save(z_map, os.path.join(output_dir, f'{var}_{params.mask}_z_map{add_info}.nii.gz'))
    plot_z_map(z_map, var, plot_path, RESOLUTION, add_info)

if RUN_PERMUTATION:

    # for var in variables_long:
    for var in params.latent_vars_long:

        eff_size_files = []
        for sub in subjects:
            #get contrast data (beta estimates) and concatinate
            ef_size = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{var}_{params.mask}_effect_size_map{add_info}.nii.gz'))
            eff_size_files.append(ef_size)

        #second level model:
        design_matrix = pd.DataFrame([1] * len(eff_size_files),
                                    columns=['intercept']) #one sample test
        perm_dict = \
            non_parametric_inference(second_level_input=eff_size_files,
                                    design_matrix=design_matrix,
                                    model_intercept=True,
                                    n_perm=N_PERM,
                                    two_sided_test=TWO_SIDED,
                                    n_jobs=N_JOBS,
                                    smoothing_fwhm=FWHM,
                                    threshold=TRESH)

        nib.save(perm_dict['logp_max_t'],
                        os.path.join(output_dir,
                                    f'{var}_logp_max_t{add_info}_{FWHM}.nii.gz'))
        nib.save(perm_dict['logp_max_size'],
                        os.path.join(output_dir,
                                    f'{var}_logp_max_size{add_info}_{FWHM}.nii.gz'))
        nib.save(perm_dict['logp_max_mass'],
                    os.path.join(output_dir,
                                f'{var}_logp_max_mass{add_info}_{FWHM}.nii.gz'))
        
