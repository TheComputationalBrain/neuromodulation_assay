#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:32:01 2024

This creates a receptors density matrix from the PET data made avaiable by Hansen et al (2022). 
As suggested, this script only takes the tracer with the best binding potential (if multiple tracers are available for a receptor)
and creates a weighted average if there are multiple datasets with the same tracer. 
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import pickle 
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
import seaborn as sns
import fmri_funcs as fun
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn import image, plotting, datasets, surface
from neuromaps import transforms
import cmcrameri.cm as cmc
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import main_funcs as mf
from config.loader import load_config


PLOT_RECEPTORS=False
PLOT_RECEPTORS_INFLATED=False
PLOT_CORR = False

params, paths, rec = load_config('all', return_what='all')

output_dir = os.path.join(paths.home_dir,'receptors', 'PET2')
os.makedirs(output_dir, exist_ok = True)

if params.parcelated == False:
     masker = fun.get_masker(params=params, paths=paths)
elif (params.mask == 'schaefer') & params.parcelated:
    atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #remove everything labeled as background
else:
    raise ValueError("Unknown atlas!")

masker.fit()

if rec.source == "PET2":

    receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                            "MOR", "NET", "NMDA", "VAChT", "A2"])

    receptors_nii = [paths.receptor_path + '/5HT1a_way_hc36_savli.nii',
                    paths.receptor_path + '/5HT1b_p943_hc22_savli.nii',
                    paths.receptor_path + '/5HT1b_p943_hc65_gallezot.nii.gz',
                    paths.receptor_path + '/5HT2a_cimbi_hc29_beliveau.nii',
                    paths.receptor_path + '/5HT4_sb20_hc59_beliveau.nii',
                    paths.receptor_path + '/5HT6_gsk_hc30_radhakrishnan.nii.gz',
                    paths.receptor_path + '/5HTT_dasb_hc100_beliveau.nii',
                    paths.receptor_path + '/A4B2_flubatine_hc30_hillmer.nii.gz',
                    paths.receptor_path + '/CB1_omar_hc77_normandin.nii.gz',
                    paths.receptor_path + '/D1_SCH23390_hc13_kaller.nii',
                    paths.receptor_path + '/D2_flb457_hc37_smith.nii.gz',
                    paths.receptor_path + '/D2_flb457_hc55_sandiego.nii.gz',
                    paths.receptor_path + '/DAT_fpcit_hc174_dukart_spect.nii',
                    paths.receptor_path + '/GABAa-bz_flumazenil_hc16_norgaard.nii',
                    paths.receptor_path + '/H3_cban_hc8_gallezot.nii.gz', 
                    paths.receptor_path + '/M1_lsn_hc24_naganawa.nii.gz',
                    paths.receptor_path + '/mGluR5_abp_hc22_rosaneto.nii',
                    paths.receptor_path + '/mGluR5_abp_hc28_dubois.nii',
                    paths.receptor_path + '/mGluR5_abp_hc73_smart.nii',
                    paths.receptor_path + '/MU_carfentanil_hc204_kantonen.nii',
                    paths.receptor_path + '/NAT_MRB_hc77_ding.nii.gz',
                    paths.receptor_path + '/NMDA_ge179_hc29_galovic.nii.gz',
                    paths.receptor_path + '/VAChT_feobv_hc4_tuominen.nii',
                    paths.receptor_path + '/VAChT_feobv_hc5_bedard_sum.nii',
                    paths.receptor_path + '/VAChT_feobv_hc18_aghourian_sum.nii',
                    paths.alpha_path + '/Mean_Yohimbine_HC2050.nii']
    
else:
    receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                            "MOR", "NET", "NMDA", "VAChT"])

    receptors_nii = [paths.receptor_path + '/5HT1a_way_hc36_savli.nii',
                    paths.receptor_path + '/5HT1b_p943_hc22_savli.nii',
                    paths.receptor_path + '/5HT1b_p943_hc65_gallezot.nii.gz',
                    paths.receptor_path + '/5HT2a_cimbi_hc29_beliveau.nii',
                    paths.receptor_path + '/5HT4_sb20_hc59_beliveau.nii',
                    paths.receptor_path + '/5HT6_gsk_hc30_radhakrishnan.nii.gz',
                    paths.receptor_path + '/5HTT_dasb_hc100_beliveau.nii',
                    paths.receptor_path + '/A4B2_flubatine_hc30_hillmer.nii.gz',
                    paths.receptor_path + '/CB1_omar_hc77_normandin.nii.gz',
                    paths.receptor_path + '/D1_SCH23390_hc13_kaller.nii',
                    paths.receptor_path + '/D2_flb457_hc37_smith.nii.gz',
                    paths.receptor_path + '/D2_flb457_hc55_sandiego.nii.gz',
                    paths.receptor_path + '/DAT_fpcit_hc174_dukart_spect.nii',
                    paths.receptor_path + '/GABAa-bz_flumazenil_hc16_norgaard.nii',
                    paths.receptor_path + '/H3_cban_hc8_gallezot.nii.gz', 
                    paths.receptor_path + '/M1_lsn_hc24_naganawa.nii.gz',
                    paths.receptor_path + '/mGluR5_abp_hc22_rosaneto.nii',
                    paths.receptor_path + '/mGluR5_abp_hc28_dubois.nii',
                    paths.receptor_path + '/mGluR5_abp_hc73_smart.nii',
                    paths.receptor_path + '/MU_carfentanil_hc204_kantonen.nii',
                    paths.receptor_path + '/NAT_MRB_hc77_ding.nii.gz',
                    paths.receptor_path + '/NMDA_ge179_hc29_galovic.nii.gz',
                    paths.receptor_path + '/VAChT_feobv_hc4_tuominen.nii',
                    paths.receptor_path + '/VAChT_feobv_hc5_bedard_sum.nii',
                    paths.receptor_path + '/VAChT_feobv_hc18_aghourian_sum.nii']

for proj in ['vol', 'surf']:  
    masked = []
    for receptor in receptors_nii:
        img = nib.load(receptor)
        if proj == 'vol':
            masked.append(masker.fit_transform(img))
        else:
            data_surf = transforms.mni152_to_fsaverage(img, fsavg_density='41k')
            data_gii = []
            for img in data_surf:
                data_hemi = img.agg_data()
                data_hemi = np.asarray(data_hemi).T
                data_gii.append(data_hemi)
            receptor_array = np.hstack(data_gii).reshape(1,-1)
            masked.append(receptor_array) 
        
    r = np.zeros([masked[0].shape[1], len(masked)])
    for i in range(len(masked)):
        r[:, i] = masked[i] 

    receptor_data = np.zeros([r.shape[0], len(receptor_names)])
    receptor_data[:, 0] = r[:, 0]
    receptor_data[:, 2:9] = r[:, 3:10]
    receptor_data[:, 10:14] = r[:, 12:16]
    receptor_data[:, 15:18] = r[:, 19:22]

    if rec.source == 'PET2':
        receptor_data[:, 19] = r[:, 25]

    # weighted average of 5HT1B p943
    receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

    # weighted average of D2 flb457
    receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

    # weighted average of mGluR5 ABP688
    receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

    # weighted average of VAChT FEOBV
    receptor_data[:, 18] = (zscore(r[:, 22])*4 + zscore(r[:, 23])*5 + zscore(r[:, 24])*18) / (4+5+18)

    #save receptor density maps 
    if proj == 'vol':
        if params.parcelated:
            with open(os.path.join(output_dir, 
                            f'receptor_density_{params.mask}_{params.mask_details}.pickle'), 'wb') as f:
                    pickle.dump(receptor_data, f)
        else:
            with open(os.path.join(output_dir, 
                                f'receptor_density.pickle'), 'wb') as f:
                    pickle.dump(receptor_data, f)
    else:
        if params.parcelated:
            with open(os.path.join(output_dir, 
                            f'receptor_density_{params.mask}_{params.mask_details}_surf.pickle'), 'wb') as f:
                    pickle.dump(receptor_data, f)
        else:
            with open(os.path.join(output_dir, 
                                f'receptor_density_surf.pickle'), 'wb') as f:
                    pickle.dump(receptor_data, f)


#### plotting   

#plot surface maps 
if PLOT_RECEPTORS:
    if params.parcelated:
        plot_path = os.path.join(output_dir, params.mask_details,'figures') 
    else:
        plot_path = os.path.join(output_dir,'figures') 
    if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
    for indx,receptor in enumerate(receptor_names):
        data = masker.inverse_transform(zscore(receptor_data[:, indx]))
        mask_img = image.new_img_like(data, image.get_data(data) != 0) 
        plotting.plot_img_on_surf(data, surf_mesh='fsaverage5', threshold=1e-50, bg_on_data=False,
                                        hemispheres=['left', 'right'], views=['lateral', 'medial'],
                                        title=receptor, colorbar=True, cmap = 'plasma', symmetric_cbar=False)
        fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
        plt.savefig(os.path.join(plot_path, fig_fname),dpi=300, bbox_inches='tight',transparent=True)
        plt.close()

if PLOT_RECEPTORS_INFLATED:
    mf.set_publication_style(font_size=7, layout="3-across")

    if params.parcelated:
        plot_path = os.path.join(output_dir, params.mask_details,'figures') 
    else:
        plot_path = os.path.join(output_dir,'figures') 
    if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
    for indx,receptor in enumerate(receptor_names):
        data = masker.inverse_transform(zscore(receptor_data[:, indx]))
        mask_img = image.new_img_like(data, image.get_data(data) != 0) 

        plotting.plot_img_on_surf(data, surf_mesh='fsaverage',threshold=1e-50,
                                        hemispheres=['right'], views=['lateral'],
                                        title=receptor, colorbar=True, cmap = 'plasma',inflate=True, symmetric_cbar=False)
        for ax in plt.gcf().axes:
            for coll in ax.collections:
                coll.set_rasterized(True)
        fig_fname = 'surface_receptor_'+receptor+'_cortical_right.pdf'
        plt.savefig(os.path.join(plot_path, fig_fname),dpi=300, bbox_inches='tight',transparent=True)
        plt.close()


if PLOT_CORR:
    mf.set_publication_style(font_size=7, layout="2-across")

    #### correlation matrix
    serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
    acetylcholine = ["A4B2", "M1", "VAChT"]
    noradrenaline = ["NET", "A2"]
    opioid = ["MOR"]
    glutamate = ["mGluR5", "NMDA"]
    histamine = ["H3"]
    gaba = ["GABAa"]
    dopamine = ["D1", "D2", "DAT"]
    cannabinnoid = ["CB1"]
    receptor_groups = [serotonin, acetylcholine, noradrenaline, opioid, glutamate, histamine, gaba, dopamine, cannabinnoid]

    cmap = mf.get_custom_colormap(map_type='diverging')

    ordered_receptors = [receptor for group in receptor_groups for receptor in group]
    df = pd.DataFrame(receptor_data, columns=receptor_names)
    df = df.apply(zscore)
    df = df[ordered_receptors]
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap=cmap, vmin=-1, vmax=1, linewidths=0.6, square=True)
    current_pos = 0
    for group in receptor_groups:
        group_size = len(group)
        plt.gca().add_patch(Rectangle((current_pos, current_pos), group_size, group_size, fill=False, edgecolor='black', lw=2))
        current_pos += group_size

    plot_path = os.path.join(output_dir,'figures_inflated') 
    if params.parcelated:
        fig_fname = f'receptor_corr_matrix_{params.mask}_{params.mask_details}.png'
        plt.savefig(os.path.join(plot_path, fig_fname))
    else:
        fig_fname = f'receptor_corr_matrix_{params.mask}.pdf'
        plt.savefig(os.path.join(plot_path, fig_fname))




