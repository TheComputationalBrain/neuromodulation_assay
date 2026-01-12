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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import nibabel as nib
from neuromaps import transforms
from scipy.stats import zscore
from nilearn import image, datasets, masking


def set_publication_style(
    font_size=7,
    line_width=1,
    context="paper",
    layout="single",
    page="full",
):
    """
    Set consistent, publication-quality figure style 
    
    Parameters
    ----------
    font_size : int
        Base font size for all text.
    line_width : float
        Default line width for axes and lines.
    context : str
        Seaborn context ('paper', 'notebook', 'talk', 'poster').
    layout : str
        'single', '2-across', '3-across', or '6-across'
    page : str
        'single' (column width ≈ 8.9 cm) or 'full' (page width ≈ 17.8 cm)
    """

    import matplotlib
    import seaborn as sns

    cm_to_inch = 1 / 2.54

    # --- PNAS-style page widths ---
    page_widths = {
        "single": 8.9 * cm_to_inch,   # ≈3.5"
        "full": 17.8 * cm_to_inch,    # ≈7.0"
    }

    if page not in page_widths:
        raise ValueError("page must be 'single' or 'full'")

    page_width = page_widths[page]
    gutter = 0.25 * cm_to_inch  # ≈0.1" between panels

    # --- Compute figure width ---
    n_panels = {
        "single": 1,
        "2-across": 2,
        "3-across": 3,
        "6-across": 6,
    }.get(layout)

    if n_panels is None:
        raise ValueError("layout must be 'single', '2-across', '3-across', or '6-across'")

    total_gutter = (n_panels - 1) * gutter
    fig_width = (page_width - total_gutter) / n_panels

    # Aspect ratio — for 6-across, use a shallower height
    if layout == "6-across":
        fig_height = fig_width * 0.65
    else:
        fig_height = fig_width * 0.75

    # Error bar cap scaling
    capsize = {
        "single": 2.5,
        "2-across": 2.0,
        "3-across": 1.5,
        "6-across": 1.0,
    }[layout]
    matplotlib.rcParams["errorbar.capsize"] = capsize

    # --- Apply styling ---
    sns.set_theme(
        style="ticks",
        context=context,
        font="sans-serif",
        rc={
            "figure.figsize": (fig_width, fig_height),
            "figure.dpi": 300,

            # Fonts
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,

            # Lines & ticks
            "axes.linewidth": 0.5,
            "lines.linewidth": line_width,
            "lines.markersize": 3,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,

            # Export-friendly
            "savefig.transparent": True,
        },
    )

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'


def save_figure(fig, output_dir, filename, extension=['pdf', 'svg'], close=True):
    """
    Apply publication cleanup (tight layout, despine) and save figure in SVG and PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.despine(trim=False)
    fig.tight_layout() 

    base_path = os.path.join(output_dir, filename)
    svg_path = f"{base_path}.svg"
    pdf_path = f"{base_path}.pdf"

    if 'svg' in extension:
        fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    if 'pdf' in extension:
        fig.savefig(pdf_path, bbox_inches="tight", transparent=True)

    if close:
        plt.close(fig)

def get_custom_colormap(map_type="diverging", N=256):
    """
    Returns a perceptually balanced colormap.

    Parameters
    ----------
    map_type : str
        One of {"diverging", "pos", "neg"}:
          - "diverging": dark blue → light blue → white → light orange → dark red
          - "neg": dark blue → light blue → white
          - "pos": white → light orange → dark red
    N : int
        Number of colors in the map.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
    """

    # Define 3-point segments for each side
    neg_colors = ["#08306B", "#4292C6", "#FFFFFF"]
    pos_colors = ["#FFFFFF", "#F7AD45", "#8C1007"]

    if map_type == "diverging":
        # Create negative and positive halves
        n_half = N // 2
        cmap_neg = LinearSegmentedColormap.from_list("neg_half", neg_colors, N=n_half)
        cmap_pos = LinearSegmentedColormap.from_list("pos_half", pos_colors, N=n_half)

        # Sample both halves and concatenate
        neg_part = cmap_neg(np.linspace(0, 1, n_half))
        pos_part = cmap_pos(np.linspace(1/N, 1, n_half))  # slight offset to avoid duplicate white
        colors_full = np.vstack((neg_part, pos_part))

        cmap = LinearSegmentedColormap.from_list("diverging_balanced", colors_full, N=N)

    elif map_type == "neg":
        cmap = LinearSegmentedColormap.from_list("blue_white", neg_colors, N=N)

    elif map_type == "pos":
        cmap = LinearSegmentedColormap.from_list("white_red", pos_colors, N=N)

    else:
        raise ValueError("map_type must be one of {'diverging', 'pos', 'neg'}.")

    return cmap


def demean(x):
    """center regressors"""
    mean = np.nanmean(x)
    return x - mean

def get_json_dir(db_name, paths):
   
    json_files_dir = {'NAConf': op.join(paths.root_dir, paths.data_dir, 'bids_dataset'),
                    'EncodeProb': op.join(paths.root_dir, paths.data_dir, 'bids_dataset'),
                    'Explore': op.join(paths.root_dir, paths.data_dir, 'bids/raw/'),
                    'PNAS': op.join(paths.root_dir, paths.data_dir, 'MRI_data/analyzed_data')}
    
    return json_files_dir[db_name]


def get_fmri_dir(db_name, paths):
    
    fmri_dir = {'NAConf': op.join(paths.root_dir, paths.data_dir, 'derivatives'),
                'EncodeProb': op.join(paths.root_dir, paths.data_dir, 'derivatives'),
                'Explore': op.join(paths.root_dir, paths.data_dir, 'bids/derivatives/fmriprep-23.1.3_MAIN'),
                'PNAS': op.join(paths.root_dir, paths.data_dir, 'MRI_data/analyzed_data')}

    return fmri_dir[db_name]


def get_beh_dir(db_name, paths):

    beh_dir = {'NAConf': op.join(paths.root_dir, paths.data_dir),
               'EncodeProb': op.join(paths.root_dir, paths.data_dir),
               'Explore': '/home_local/EXPLORE/github/explore/2021_Continous_2armed',
               'PNAS': op.join(paths.root_dir, paths.data_dir)}

    return beh_dir[db_name]

def get_subjects(db, data_dir): 
    subjects = []

    if db == 'lanA':
        subjects = list(range(400, 461))
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

def get_beta_dir_and_info(task, params, paths):

    beta_dir  = os.path.join(paths.home_dir,task,'first_level') 

    if task == 'NAConf':
        add_info = '_firstTrialsRemoved'
    elif not params.zscore_per_session:
        add_info = '_zscoreAll'
    else:
        add_info = ""

    return beta_dir, add_info

def load_effect_map_array(sub, task, latent_var, params, paths):
    beta_dir, add_info = get_beta_dir_and_info(task, params, paths)

    map = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()

    return map

def load_receptor_array(params, paths, rec, on_surface):
    if on_surface:
        receptor_data =zscore(np.load(os.path.join(paths.home_dir,'receptors', rec.source,f'receptor_density_surf.pickle'), allow_pickle=True))
    else:
        receptor_data = zscore(np.load(os.path.join(paths.home_dir,'receptors', rec.source,f'receptor_density.pickle'), allow_pickle=True))

    return receptor_data

def load_surface_effect_maps_for_cv(subjects, task, latent_var, params, paths):
    beta_dir, add_info = get_beta_dir_and_info(task, params, paths)

    if task == 'lanA':
        fmri_files = []
        for subj in subjects:
            subj_id = f"{subj:03d}"  
            pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
            fmri_files.extend(glob.glob(pattern))
    else:
        fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}__effect_size_map{add_info}.nii.gz')))
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


def nii_to_cortical_voxel_array(
    nii_path,
    output_dir,
    schaefer_n_rois=100,
    save=True
):
    """
    Convert a .nii.gz effect map into a voxel-wise cortical array

    Parameters
    ----------
    nii_path : str
        Path to the .nii.gz effect map.
    output_dir : str
        Directory to save the output array.
    save : bool
        Whether to save the array as .npy.

    Returns
    -------
    voxel_array : np.ndarray
        1D array of cortical voxel values.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load effect map
    effect_img = image.load_img(nii_path)

    # Load Schaefer atlas
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=schaefer_n_rois,
        resolution_mm=2
    )

    atlas_img = image.load_img(schaefer.maps)

    # Create a binary cortical mask (all parcels > 0)
    cortical_mask = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #mask background

    # Resample effect map to mask if needed: this is the size of the receptor data provided
    cortical_mask = image.resample_to_img(
        effect_img, cortical_mask, interpolation="nearest"
    )

    # Apply mask → 1D voxel array
    voxel_array = masking.apply_mask(effect_img, cortical_mask)


    return voxel_array, cortical_mask


