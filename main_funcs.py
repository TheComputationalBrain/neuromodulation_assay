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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import nibabel as nib
from neuromaps import transforms
from scipy.stats import zscore

def set_publication_style(font_size=7, line_width=1, context="paper", layout="single"):
    """
    Set consistent, publication-quality figure style.
    
    Parameters
    ----------
    font_size : int
        Base font size for all text.
    line_width : float
        Default line width for axes and lines.
    context : str
        Seaborn plotting context ('paper', 'notebook', 'talk', 'poster').
    layout : str
        Publication layout: 'single', '2-across', or '3-across'
        - single: one plot per column (3.35" width)
        - 2-across: two plots spanning page width (≈3.3" each)
        - 3-across: three plots spanning page width (≈2.15" each)
    """

    # Choose figure width based on layout
    if layout == "single":
        figsize = (3.35, 2.6)  # 85 mm width
        capsize = 2.5
    elif layout == "2-across":
        figsize = (3.3, 2.5)   # half of double-column width
        capsize = 2.0
    elif layout == "3-across":
        figsize = (2.15, 2.3)  # one-third of double-column width
        capsize = 1.5
    else:
        raise ValueError("layout must be: 'single', '2-across', or '3-across'")
    
    matplotlib.rcParams['errorbar.capsize'] = capsize

    sns.set_theme(
        style="ticks",
        context=context,
        font="sans-serif",
        rc={
            # Figure sizing
            "figure.figsize": figsize,
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

    # Embed fonts correctly in vector exports
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

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
    pos_colors = ["#FFFFFF", "#F7AD45", "#BB3E00"]

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

def get_json_dir(db_name, root_dir, data_dir):
   
    json_files_dir = {'NAConf': op.join(root_dir, data_dir, 'bids_dataset'),
                    'EncodeProb': op.join(root_dir, data_dir, 'bids_dataset'),
                    'Explore': op.join(root_dir, data_dir, 'bids/raw/'),
                    'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}
    
    return json_files_dir[db_name]


def get_fmri_dir(db_name, root_dir, data_dir):
    
    fmri_dir = {'NAConf': op.join(root_dir, data_dir, 'derivatives'),
                'EncodeProb': op.join(root_dir, data_dir, 'derivatives'),
                'Explore': op.join(root_dir, data_dir, 'bids/derivatives/fmriprep-23.1.3_MAIN'),
                'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}

    return fmri_dir[db_name]


def get_beh_dir(db_name, root_dir, data_dir):

    beh_dir = {'NAConf': op.join(root_dir, data_dir),
               'EncodeProb': op.join(root_dir, data_dir),
               'Explore': '/home_local/EXPLORE/github/explore/2021_Continous_2armed',
               'PNAS': op.join(root_dir, data_dir)}

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

def get_beta_dir_and_info(task, params,paths):
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

def load_effect_map_array(sub, task, latent_var, params, paths):
    beta_dir, add_info = get_beta_dir_and_info(task, params, paths)

    map = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{params.mask}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()

    return map

def load_receptor_array(params, paths, rec, on_surface):
    if on_surface:
        receptor_data =zscore(np.load(os.path.join(paths.home_dir,'receptors', rec.source,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
    else:
        receptor_data = zscore(np.load(os.path.join(paths.home_dir,'receptors', rec.source,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))

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




