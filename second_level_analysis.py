#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alice Hodapp 

This script computes group level estimates based on first-level
contrasts and plots the results. First-level contrasts have been parformed seperatly. 

"""

import os
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
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting, image
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from sklearn.linear_model import LinearRegression
import main_funcs as mf
import nibabel as nib
from scipy.stats import zscore
from params_and_paths import Paths, Params, Receptors
from dominance_stats import dominance_stats
from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_atlas_schaefer_2018
from multiprocessing import Pool


FROM_OLS = False

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

output_dir = os.path.join(paths.home_dir,params.db,params.mask,'second_level')
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 
plot_path = os.path.join(output_dir, 'plot_raw')
if not os.path.exists(plot_path):
    os.makedirs(plot_path) 

#get receptor data
if FROM_OLS:
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'OLS')
else: 
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level')

if params.parcelated:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
    mask_comb = params.mask + '_' + params.mask_details 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
else:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', 'PET') #vertex level analyis can only be run on PET data densities 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
    mask_comb = params.mask 

if rec.source == 'autorad_zilles44':
    #autoradiography dataset is only one hemisphere 
    receptor_density = np.concatenate((receptor_density, receptor_density))

if params.mask == 'harvard_oxford_cortical':
    atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    parcelated = False
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #all cortical areas
    masker = NiftiMasker(mask_img=mask_img)
elif params.mask == 'harvard_oxford_subcortical':
    atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    parcelated = False
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #all subcortical areas
    masker = NiftiMasker(mask_img=mask_img)
elif params.mask == 'schaefer':
    parcelated = True
    atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #remove everything labeled as background
elif params.mask == 'schaefer_byvoxel':
    parcelated = False
    atlas = fetch_atlas_schaefer_2018(n_rois=int(100), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0)  #remove everything labeled as background
    masker = NiftiMasker(mask_img=mask_img)
else:
    raise ValueError("Unknown atlas!")

def run_lin_reg(X,y):
    results_dict = dict()
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    results_dict['coefs'] = lin_reg.coef_

    #adjusted R2
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    results_dict['r_squared'] = r_squared
    results_dict['adjusted_r_squared'] = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    
    return results_dict

def plot_res(res, r2):
    plt.rcParams.update({'font.size': 16})

    if rec.source == 'PET':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.dopamine, rec.cannabinnoid]
    elif rec.source  == 'autorad_zilles44':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.glutamate, rec.gaba, rec.dopamine]
    receptor_class = [rec.exc,rec.inh]

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
    ax.set_xticklabels(ordered_receptors, rotation=90)
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

def run_group_analysis(latent_var):

    print(f'------- running group analysis for {latent_var} -------')

    eff_size_files = []
    for sub in subjects:
        #get contrast data (beta estimates) and concatinate
        ef_size = nib.load(os.path.join(paths.home_dir,
                                                params.db,
                                                params.mask,
                                                'first_level',
                                                f'sub-{sub:02d}_{latent_var}_{params.mask}_effect_size_map.nii.gz'))
        eff_size_files.append(ef_size)

    #second level model:
    design_matrix = pd.DataFrame([1] * len(eff_size_files),
                                 columns=['intercept']) #one sample test
    second_level_model = SecondLevelModel(smoothing_fwhm=params.smoothing_fwhm)
    second_level_model = second_level_model.fit(eff_size_files,
                                                design_matrix=design_matrix)
    
    #second level contrast
    #tscore
    t_map = second_level_model.compute_contrast(output_type='stat')
    fname = f'{latent_var}_{params.mask}_tmap.nii.gz'
    nib.save(t_map, os.path.join(output_dir, fname))
    plotting.plot_img_on_surf(t_map, surf_mesh='fsaverage5', 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=latent_var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True)

    fname = f'{latent_var}_tmap.png' 
    plt.savefig(os.path.join(plot_path, fname))

    #as np array 
    y_data = masker.fit_transform(t_map).flatten()

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

    print('-- running regression for tmap--')
    reg_results = run_lin_reg(X,y)
    fname = f'tmap_{latent_var}_group_reg.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(reg_results, f)

    #dominance analysis
    print('-- running dominance for tmap--')
    dominance_results = dominance_stats(X, y)
    fname = f'tmap_{latent_var}_group_da.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(dominance_results, f)
    fig = plot_res(dominance_results['total_dominance'], dominance_results['full_r_sq'])
    fname = f'tmap_{latent_var}_group_da.png' 
    fig.savefig(os.path.join(plot_path, fname))

    #mean beta
    plt.rcParams.update({'font.size': 8})
    effect_map = second_level_model.compute_contrast(output_type='effect_size')
    fname = f'{latent_var}_{params.mask}_effect_map.nii.gz'
    nib.save(effect_map, os.path.join(output_dir, fname))
    plotting.plot_img_on_surf(effect_map, surf_mesh='fsaverage5', 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=latent_var, colorbar=True, cmap = 'cold_hot',inflate=True, symmetric_cbar=True, cbar_tick_format='%.2f')
    fname = f'{latent_var}_effect_map.png' 
    plt.savefig(os.path.join(plot_path, fname))

    #as np array 
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

    print('-- running regression for beta--')
    reg_results = run_lin_reg(X,y)
    fname = f'effect_{latent_var}_group_reg.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(reg_results, f)

    #dominance analysis
    print('-- running dominance for beta--')
    dominance_results = dominance_stats(X, y)
    fname = f'effect_{latent_var}_group_da.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(dominance_results, f)
    fig = plot_res(dominance_results['total_dominance'], dominance_results['full_r_sq'])
    fname = f'effect_{latent_var}_group_da.png' 
    fig.savefig(os.path.join(plot_path, fname))


    #variance of effect
    plt.rcParams.update({'font.size': 6})
    var_map = second_level_model.compute_contrast(output_type='effect_variance')
    fname = f'{latent_var}_{params.mask}_variance_map.nii.gz'
    nib.save(var_map, os.path.join(output_dir, fname))
    plotting.plot_img_on_surf(var_map, surf_mesh='fsaverage5', 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=latent_var, colorbar=True, cmap = 'black_red',inflate=True, symmetric_cbar='auto', cbar_tick_format='%.4f')
    fname = f'{latent_var}_variance_map.png' 
    plt.savefig(os.path.join(plot_path, fname))

    y_data = masker.fit_transform(var_map).flatten()
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

    print('-- running regression for variance--')
    reg_results = run_lin_reg(X,y)
    fname = f'variance_{latent_var}_group_reg.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(reg_results, f)

    #dominance analysis
    print('-- running dominance for variance--')
    dominance_results = dominance_stats(X, y)
    fname = f'variance_{latent_var}_group_da.pickle' 
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(dominance_results, f)
    fig = plot_res(dominance_results['total_dominance'], dominance_results['full_r_sq'])
    fname = f'variance_{latent_var}_group_da.png' 
    fig.savefig(os.path.join(plot_path, fname))


my_pool = Pool(4)
my_pool.map(run_group_analysis,params.latent_vars)

