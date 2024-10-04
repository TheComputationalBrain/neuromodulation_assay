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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import seaborn as sns
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_atlas_schaefer_2018
from nilearn import image, plotting
from params_and_paths import Paths, Params


PLOT_RECEPTORS=True
PLOT_RECEPTORS_INFLATED=False

paths = Paths()
params = Params()

#TODO: move path to paths class 
receptor_path = '/home/ah278717/hansen_receptors/data/PET_nifti_images/' #path to downloaded data from Hansen et al. (2022)
alpha_path = '/home/ah278717/alpha2_receptor/' #path to the data shared by Benedicte Ballanger
output_dir = os.path.join(paths.home_dir,'receptors', 'PET2')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if params.mask == 'harvard_oxford_cortical':
    atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #all cortical areas
    masker = NiftiMasker(mask_img=mask_img)
elif params.mask == 'harvard_oxford_subcortical':
    atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #all subcortical areas
    masker = NiftiMasker(mask_img=mask_img)
elif (params.mask == 'schaefer') & params.parcelated:
    atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #remove everything labeled as background
elif (params.mask == 'schaefer'):
    atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "Background")
    atlas_img = image.load_img(atlas.maps)
    mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #remove everything labeled as background
    masker = NiftiMasker(mask_img=mask_img)
else:
    raise ValueError("Unknown atlas!")

masker.fit()


receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT", "a2"])

receptors_nii = [receptor_path + '/5HT1a_way_hc36_savli.nii',
                 receptor_path + '/5HT1b_p943_hc22_savli.nii',
                 receptor_path + '/5HT1b_p943_hc65_gallezot.nii.gz',
                 receptor_path + '/5HT2a_cimbi_hc29_beliveau.nii',
                 receptor_path + '/5HT4_sb20_hc59_beliveau.nii',
                 receptor_path + '/5HT6_gsk_hc30_radhakrishnan.nii.gz',
                 receptor_path + '/5HTT_dasb_hc100_beliveau.nii',
                 receptor_path + '/A4B2_flubatine_hc30_hillmer.nii.gz',
                 receptor_path + '/CB1_omar_hc77_normandin.nii.gz',
                 receptor_path + '/D1_SCH23390_hc13_kaller.nii',
                 receptor_path + '/D2_flb457_hc37_smith.nii.gz',
                 receptor_path + '/D2_flb457_hc55_sandiego.nii.gz',
                 receptor_path + '/DAT_fpcit_hc174_dukart_spect.nii',
                 receptor_path + '/GABAa-bz_flumazenil_hc16_norgaard.nii',
                 receptor_path + '/H3_cban_hc8_gallezot.nii.gz', 
                 receptor_path + '/M1_lsn_hc24_naganawa.nii.gz',
                 receptor_path + '/mGluR5_abp_hc22_rosaneto.nii',
                 receptor_path + '/mGluR5_abp_hc28_dubois.nii',
                 receptor_path + '/mGluR5_abp_hc73_smart.nii',
                 receptor_path + '/MU_carfentanil_hc204_kantonen.nii',
                 receptor_path + '/NAT_MRB_hc77_ding.nii.gz',
                 receptor_path + '/NMDA_ge179_hc29_galovic.nii.gz',
                 receptor_path + '/VAChT_feobv_hc4_tuominen.nii',
                 receptor_path + '/VAChT_feobv_hc5_bedard_sum.nii',
                 receptor_path + '/VAChT_feobv_hc18_aghourian_sum.nii',
                 alpha_path + '/Mean_Yohimbine_HC2050.nii']

masked = []
for receptor in receptors_nii:
    img = nib.load(receptor)
    masked.append(masker.fit_transform(img))

r = np.zeros([masked[0].shape[1], len(masked)])
for i in range(len(masked)):
    r[:, i] = masked[i] 

receptor_data = np.zeros([r.shape[0], len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:18] = r[:, 19:22]
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
if params.parcelated:
     with open(os.path.join(output_dir, 
                       f'receptor_density_{params.mask}_{params.mask_details}.pickle'), 'wb') as f:
            pickle.dump(receptor_data, f)
else:
    with open(os.path.join(output_dir, 
                        f'receptor_density_{params.mask}.pickle'), 'wb') as f:
            pickle.dump(receptor_data, f)


#### plotting   
cmap = np.genfromtxt('/home/ah278717/hansen_receptors/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

#plot surface maps 
if PLOT_RECEPTORS:
    if params.parcelated:
        plot_path = os.path.join(output_dir, params.mask_details,'figures') 
        if not os.path.exists(plot_path):
                os.makedirs(plot_path) 
        for indx,receptor in enumerate(receptor_names):

            data = masker.inverse_transform(zscore(receptor_data[:, indx]))
            mask_img = image.new_img_like(data, image.get_data(data) != 0) 
            plotting.plot_img_on_surf(data, surf_mesh='fsaverage5', mask_img=mask_img, bg_on_data=True,
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'],
                                            title=receptor, colorbar=True, cmap = 'plasma', )
            fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
            plt.savefig(os.path.join(plot_path, fig_fname))
            plt.close()

    else:
        plot_path = os.path.join(output_dir,'figures_cmap') 
        if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        for indx,receptor in enumerate(receptor_names):
            data = masker.inverse_transform(zscore(receptor_data[:, indx]))
            plotting.plot_img_on_surf(data, surf_mesh='fsaverage5', threshold=1e-50,
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'],
                                            title=receptor, colorbar=True, cmap = cmap_div, symmetric_cbar=False, bg_on_data=False)
            fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
            plt.savefig(os.path.join(plot_path, fig_fname),dpi=300, bbox_inches='tight',transparent=True)
            plt.close()

if PLOT_RECEPTORS_INFLATED:
    if params.parcelated:
        plot_path = os.path.join(output_dir, params.mask_details,'figures_inflated') 
        if not os.path.exists(plot_path):
                os.makedirs(plot_path) 
        for indx,receptor in enumerate(receptor_names):

            data = masker.inverse_transform(zscore(receptor_data[:, indx]))
            mask_img = image.new_img_like(data, image.get_data(data) != 0) 
            plotting.plot_img_on_surf(data, surf_mesh='fsaverage', mask_img=mask_img, 
                                            hemispheres=['left'], views=['lateral', 'medial'],
                                            title=receptor, colorbar=True, cmap = 'plasma',inflate=True, symmetric_cbar=False)
            fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
            plt.savefig(os.path.join(plot_path, fig_fname))
            plt.close()

    else:
        plot_path = os.path.join(output_dir,'figures_inflated') 
        if not os.path.exists(plot_path):
                os.makedirs(plot_path) 
        for indx,receptor in enumerate(receptor_names):
            data = masker.inverse_transform(zscore(receptor_data[:, indx]))
            plotting.plot_img_on_surf(data, surf_mesh='fsaverage', 
                                            hemispheres=['left'], views=['lateral', 'medial'], threshold=1e-50,
                                            title=receptor, colorbar=True, cmap = 'plasma',inflate=True, symmetric_cbar=False)
            fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
            plt.savefig(os.path.join(plot_path, fig_fname),dpi=300, bbox_inches='tight',transparent=True)
            plt.close()



#### correlation matrix
serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
acetylcholine = ["A4B2", "M1", "VAChT"]
noradrenaline = ["NET", "a2"]
opioid = ["MOR"]
glutamate = ["mGluR5", "NMDA"]
histamine = ["H3"]
gaba = ["GABAa"]
dopamine = ["D1", "D2", "DAT"]
cannabinnoid = ["CB1"]
receptor_groups = [serotonin, acetylcholine, noradrenaline, opioid, glutamate, histamine, gaba, dopamine, cannabinnoid]

cmap = np.genfromtxt('../hansen_receptors/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

#receptor_data = np.load(os.path.join(output_dir, f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True)
ordered_receptors = [receptor for group in receptor_groups for receptor in group]
df = pd.DataFrame(receptor_data, columns=receptor_names)
df = df.apply(zscore)
df = df[ordered_receptors]
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap=cmap_div, vmin=-1, vmax=1, linewidths=0.6, square=True)
plt.title(f'{params.mask}: Correlation of Receptors')
current_pos = 0
for group in receptor_groups:
    group_size = len(group)
    plt.gca().add_patch(Rectangle((current_pos, current_pos), group_size, group_size, fill=False, edgecolor='black', lw=2))
    current_pos += group_size

if params.parcelated:
    fig_fname = f'receptor_corr_matrix_{params.mask}_{params.mask_details}.png'
    plt.savefig(os.path.join(plot_path, fig_fname))
else:
    fig_fname = f'receptor_corr_matrix_{params.mask}.png'
    plt.savefig(os.path.join(plot_path, fig_fname))

