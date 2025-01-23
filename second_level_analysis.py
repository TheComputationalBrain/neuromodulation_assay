#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alice Hodapp 

This script computes group level estimates based on first-level
contrasts and plots the results. First-level contrasts have been parformed seperatly. 

"""

import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn import plotting, image
from nilearn import datasets
from nilearn import surface
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from sklearn.linear_model import LinearRegression
import main_funcs as mf
import fmri_funcs as fun
import nibabel as nib
from scipy.stats import zscore
from dominance_stats import dominance_stats
from nilearn.datasets import fetch_atlas_schaefer_2018
from params_and_paths import Paths, Params, Receptors

paths = Paths()
params = Params()
rec = Receptors()

PLOT_ONLY = True
RUN_PERMUTATION = False
#for cluster permutation:
N_PERM = 100000
N_JOBS = 4
TRESH = 0.001
FWHM = 5
TWO_SIDED = False

#resolution of the plots
resolution = 'fsaverage5' #chnage freesurfer5 to freesurfer to get high res surface plots


#adjust naming
if params.zscore_per_session:   
    zscore_info = ""
else:
     zscore_info = "_zscoreAll"

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

if params.db == 'Explore':
    output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level')
else: 
    output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level',params.model)

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

if params.parcelated:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
    mask_comb = params.mask + '_' + params.mask_details 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
else:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) #vertex level analyis can only be run on PET data densities 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
    mask_comb = params.mask 

if params.parcelated == False:
     masker = fun.get_masker()
elif (params.mask == 'schaefer') & params.parcelated:
    atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #remove everything labeled as background
else:
    raise ValueError("Unknown atlas!")

masker.fit()

#for the group dominance analysis 
def plot_res(res, r2):
    plt.rcParams.update({'font.size': 16})

    if rec.source in ['PET', 'PET2']:
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.dopamine, rec.cannabinnoid]
    elif rec.source  == 'autorad_zilles44':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.glutamate, rec.gaba, rec.dopamine]
    receptor_class = [rec.exc,rec.inh]

    receptor_label_formatted = [
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$',
    '$5\\text{-}\\mathrm{HTT}$',
    '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$',
    '$\\mathrm{M}_{\\mathrm{1}}$',
    '$\\mathrm{VAChT}$',
    '$\\mathrm{NET}$',
    '$\\mathrm{A}_{\\mathrm{2}}$',
    '$\\mathrm{MOR}$',
    '$\\mathrm{m}\\mathrm{GluR}_{\\mathrm{5}}$',
    '$\\mathrm{NMDA}$',
    '$\\mathrm{H}_{\\mathrm{3}}$',
    '$\\mathrm{GABA}_{\\mathrm{a}}$',
    '$\\mathrm{D}_{\\mathrm{1}}$',
    '$\\mathrm{D}_{\\mathrm{2}}$',
    '$\\mathrm{DAT}$',
    '$\\mathrm{CB}_{\\mathrm{1}}$'
]

    receptor_to_group = {}
    for group_idx, group in enumerate(receptor_groups):
        for receptor in group:
            receptor_to_group[receptor] = group_idx

    ordered_receptors = [receptor for group in receptor_groups for receptor in group]

    receptor_to_class = {}
    for class_idx, class_group in enumerate(receptor_class):
        for receptor in class_group:
            receptor_to_class[receptor] = class_idx

    # Assign colors to each group
    base_colors = sns.color_palette('husl', len(receptor_groups))
    colors = []
    for receptor in ordered_receptors:
        group_idx = receptor_to_group.get(receptor, -1)
        class_type = receptor_to_class.get(receptor, -1)
        if class_type == 0:  # Excitatory
            color = sns.dark_palette(base_colors[group_idx], n_colors=3)[1]
            colors.append({'face': color, 'edge': color})
        elif class_type == 1:  # Inhibitory
            color = sns.light_palette(base_colors[group_idx], n_colors=3)[1]
            colors.append({'face': color, 'edge': color})
        else:
            face_color = sns.light_palette(base_colors[group_idx], n_colors=3)[0]
            edge_color = sns.dark_palette(base_colors[group_idx], n_colors=3)[2]
            colors.append({'face': face_color, 'edge': edge_color})

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_data = pd.DataFrame({
    'receptor': rec.receptor_names,
    'res': res/np.sum(res) #standadize by model fit
    })

    bar_data['receptor'] = pd.Categorical(bar_data['receptor'], categories=ordered_receptors, ordered=True)
    # Sorting the DataFrame by 'receptor'
    bar_data.sort_values('receptor', inplace=True)

    bars = ax.bar(bar_data['receptor'], bar_data['res'], 
                color=[color['face'] for color in colors], edgecolor=[color['edge'] for color in colors], capsize=5)

    # Manually add hatch patterns
    for i, (receptor, color) in enumerate(zip(ordered_receptors, colors)):
        if receptor not in receptor_class[0] and receptor not in receptor_class[1]:
            bars[i].set_hatch('//')

    ax.set_xticks(np.arange(len(ordered_receptors)))
    ax.set_xticklabels(receptor_label_formatted, rotation=90)
    for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
        group_idx = receptor_to_group.get(receptor, -1)
        label.set_color(base_colors[group_idx])
    ax.set_xlabel('Receptor/Transporter')
    ax.set_ylabel('contribution (%)')

    # Add mean and standard deviation of R² to the plot
    textstr = f'Mean R²: {r2:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.tight_layout()

    return fig

if params.db == 'Explore':
    variables = ['surprise', 'confidence', 'prediction']
else:
    variables = params.latent_vars

for var in params.latent_vars:

    print(f'------- running group analysis for {var} -------')

    #get contrast data (beta estimates) and concatinate
    eff_size_files = []
    for sub in subjects:
        ef_size = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{var}_{params.mask}_effect_size_map{zscore_info}.nii.gz'))
        eff_size_files.append(ef_size)

    #second level model:
    design_matrix = pd.DataFrame([1] * len(eff_size_files),
                                 columns=['intercept']) #one sample test
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(eff_size_files,
                                        design_matrix=design_matrix)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)

    #mean beta
    plt.rcParams.update({'font.size': 8})
    effect_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='effect_size')
    #save as numpy
    effect = masker.transform(effect_map)
    with open(os.path.join(output_dir, 
                                f'group_{var}_{params.mask}_effect_size{zscore_info}.pickle'), 'wb') as f:
                pickle.dump(effect, f)
    #save as nii
    fname = f'{var}_{params.mask}_effect_map{zscore_info}.nii.gz'
    nib.save(effect_map, os.path.join(output_dir, fname))
    #plotting
    plotting.plot_img_on_surf(effect_map, surf_mesh=resolution, 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True, cbar_tick_format='%.2f')
    fname = f'{var}_effect_map.png' 
    plt.savefig(os.path.join(plot_path, fname))
    #plotting right hemi only
    texture = surface.vol_to_surf(effect_map, fsaverage.pial_right)
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        colorbar=True,
        bg_map=fsaverage.sulc_right)
    fname = f'{var}_effect_map_right{zscore_info}.png' 
    plt.savefig(os.path.join(plot_path, fname), dpi=300)


    #z score
    z_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='z_score')
    #save as nii
    fname = f'{var}_{params.mask}_z_map{zscore_info}.nii.gz'
    nib.save(z_map, os.path.join(output_dir, fname))
    #plotting
    plotting.plot_img_on_surf(z_map, surf_mesh=resolution, 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True, cbar_tick_format='%.2f')
    fname = f'{var}_z_map{zscore_info}.png' 
    plt.savefig(os.path.join(plot_path, fname), dpi=300)
    #plotting right hemi only
    texture = surface.vol_to_surf(z_map, fsaverage.pial_right)
    plotting.plot_surf_stat_map(
        fsaverage.infl_right, texture, hemi='right',
        colorbar=True,
        bg_map=fsaverage.sulc_right)
    fname = f'{var}_z_map_right{zscore_info}.png' 
    plt.savefig(os.path.join(plot_path, fname))

    #dominance analysis on the group level --> exploratory, we use the subject level one for final analysis
    if PLOT_ONLY == False:
        y_data = masker.fit_transform(effect_map).flatten()
        #group level regression
        if params.parcelated:
            non_nan_region = ~np.isnan(receptor_density).any(axis=1)
            non_nan_indices = np.where(non_nan_region)[0]
            X = receptor_density[non_nan_indices,:] #manual assignment of autored data means that some regions are empty
            y = y_data[non_nan_indices]
        else:
            non_nan_indices = ~np.isnan(y_data)
            X = receptor_density[non_nan_indices,:] #non parcelated data might contain a few NaNs from voxels with constant activation 
            y = y_data[non_nan_indices]
        #dominance analysis
        dominance_results = dominance_stats(X, y)
        fname = f'beta_{var}_group_da{zscore_info}.pickle' 
        with open(os.path.join(output_dir, fname), 'wb') as f:
            pickle.dump(dominance_results, f)
        fig = plot_res(dominance_results['total_dominance'], dominance_results['full_r_sq'])
        fname = f'beta_{var}_group_da{zscore_info}.png' 
        fig.savefig(os.path.join(plot_path, fname))

if RUN_PERMUTATION:

    if params.db == 'Explore':
        variables_long = ['surprise', 'confidence', 'prediction','surprise_neg', 'confidence_neg', 'prediction_neg']
    else:
        variables_long = params.latent_vars_long

    for var in variables_long:
        eff_size_files = []
        for sub in subjects:
            #get contrast data (beta estimates) and concatinate
            ef_size = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{var}_{params.mask}_effect_size_map{zscore_info}.nii.gz'))
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
                                    f'{var}_logp_max_t{zscore_info}_{FWHM}.nii.gz'))
        nib.save(perm_dict['logp_max_size'],
                        os.path.join(output_dir,
                                    f'{var}_logp_max_size{zscore_info}_{FWHM}.nii.gz'))
        nib.save(perm_dict['logp_max_mass'],
                    os.path.join(output_dir,
                                f'{var}_logp_max_mass{zscore_info}_{FWHM}.nii.gz'))
        
