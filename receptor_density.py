#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:32:01 2024

@author: Alice Hodapp
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import pickle 
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import image, surface, datasets, plotting
#from netneurotools import plotting
from params_and_paths import *

receptor_path = '/home/ah278717/hansen_receptors/data/PET_nifti_images/' #path to downloaded data from Hansen et al. (2020)
output_dir = os.path.join(home_dir[DATA_ACCESS],'receptors')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if (MASK == 'harvard_oxford') & (MASK_NAME == 'cortical'):
        maxprob_atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
elif (MASK == 'harvard_oxford') & (MASK_NAME == 'subcortical'):
    maxprob_atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
else:
    raise ValueError("Unknown atlas!")

atlas_img = image.load_img(maxprob_atlas['maps'])
mask_img = image.new_img_like(atlas_img, image.get_data(atlas_img) != 0) #mask background
masker = NiftiMasker(mask_img=mask_img)
masker.fit()

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])

receptors_nii = [receptor_path+'5HT1a_way_hc36_savli.nii',
                 receptor_path+'5HT1a_cumi_hc8_beliveau.nii',
                 receptor_path+'5HT1b_az_hc36_beliveau.nii',
                 receptor_path+'5HT1b_p943_hc22_savli.nii',
                 receptor_path+'5HT1b_p943_hc65_gallezot.nii.gz',
                 receptor_path+'5HT2a_cimbi_hc29_beliveau.nii',
                 receptor_path+'5HT2a_alt_hc19_savli.nii',
                 receptor_path+'5HT2a_mdl_hc3_talbot.nii.gz',
                 receptor_path+'5HT4_sb20_hc59_beliveau.nii',
                 receptor_path+'5HT6_gsk_hc30_radhakrishnan.nii.gz',
                 receptor_path+'5HTT_dasb_hc100_beliveau.nii',
                 receptor_path+'5HTT_dasb_hc30_savli.nii',
                 receptor_path+'A4B2_flubatine_hc30_hillmer.nii.gz',
                 receptor_path+'CB1_omar_hc77_normandin.nii.gz',
                 receptor_path+'CB1_FMPEPd2_hc22_laurikainen.nii',
                 receptor_path+'D1_SCH23390_hc13_kaller.nii',
                 receptor_path+'D2_fallypride_hc49_jaworska.nii',
                 receptor_path+'D2_flb457_hc37_smith.nii.gz',
                 receptor_path+'D2_flb457_hc55_sandiego.nii.gz',
                 receptor_path+'D2_raclopride_hc7_alakurtti.nii',
                 receptor_path+'DAT_fpcit_hc174_dukart_spect.nii',
                 receptor_path+'DAT_fepe2i_hc6_sasaki.nii.gz',
                 receptor_path+'GABAa-bz_flumazenil_hc16_norgaard.nii',
                 receptor_path+'GABAa_flumazenil_hc6_dukart.nii',
                 receptor_path+'H3_cban_hc8_gallezot.nii.gz',
                 receptor_path+'M1_lsn_hc24_naganawa.nii.gz',
                 receptor_path+'mGluR5_abp_hc22_rosaneto.nii',
                 receptor_path+'mGluR5_abp_hc28_dubois.nii',
                 receptor_path+'mGluR5_abp_hc73_smart.nii',
                 receptor_path+'MU_carfentanil_hc204_kantonen.nii',
                 receptor_path+'MU_carfentanil_hc39_turtonen.nii',
                 receptor_path+'NAT_MRB_hc77_ding.nii.gz',
                 receptor_path+'NAT_MRB_hc10_hesse.nii',
                 receptor_path+'NMDA_ge179_hc29_galovic.nii.gz',
                 #receptor_path+'VAChT_feobv_hc3_spreng.nii', #not availble in volumetric form yet
                 receptor_path+'VAChT_feobv_hc4_tuominen.nii',
                 receptor_path+'VAChT_feobv_hc5_bedard_sum.nii',
                 receptor_path+'VAChT_feobv_hc18_aghourian_sum.nii']

masked = []
for receptor in receptors_nii:
    img = nib.load(receptor)
    masked.append(masker.fit_transform(img))

r = np.zeros([masked[0].shape[1], len(masked)])
for i in range(len(masked)):
    r[:, i] = masked[i] 

receptor_data = np.zeros([r.shape[0], len(receptor_names)])
receptor_data[:, 0] = (zscore(r[:, 0])*35 + zscore(r[:, 1])*8) / (8+35) #5HT1a
receptor_data[:, 1] = (zscore(r[:, 2])*36 + zscore(r[:, 3])*22 + zscore(r[:, 4])*65) / (36+23+65) #5HT1b
receptor_data[:, 2] = (zscore(r[:, 5])*29 + zscore(r[:, 6])*19 + zscore(r[:, 7])*3) / (29+19+3) #5HT1b
receptor_data[:, 3] = r[:, 8] #5HT4
receptor_data[:, 4] = r[:, 9] #5HT6
receptor_data[:, 5] = (zscore(r[:, 10])*100 + zscore(r[:, 11])*8) / (100+8) #5HTT
receptor_data[:, 6] = r[:, 12] #A1B2
receptor_data[:, 7] = (zscore(r[:, 13])*77 + zscore(r[:, 14])*22) / (77+22) #CB1
receptor_data[:, 8] = r[:, 15] #D1
receptor_data[:, 9] = (zscore(r[:, 16])*49 + zscore(r[:, 17])*37 + zscore(r[:, 18])*55 + zscore(r[:, 19])*7) / (49+37+55+7) #D2
receptor_data[:, 10] = (zscore(r[:, 20])*174 + zscore(r[:, 21])*6) / (174+6) #DAT
receptor_data[:, 11] = (zscore(r[:, 22])*16 + zscore(r[:, 23])*6) / (16+6) #GABA
receptor_data[:, 12] = r[:, 24] #H3
receptor_data[:, 13] = r[:, 25] #M1
receptor_data[:, 14] = (zscore(r[:, 26])*22 + zscore(r[:, 27])*28 + zscore(r[:, 28])*73) / (22+28+73) #mGluR5
receptor_data[:, 15] = (zscore(r[:, 29])*204 + zscore(r[:, 30])*39) / (204+39) #MOR
receptor_data[:, 16] = (zscore(r[:, 31])*77 + zscore(r[:, 32])*10) / (77+10) #NET
receptor_data[:, 17] = r[:, 33] #NMDA
receptor_data[:, 18] = (zscore(r[:, 34])*4 + zscore(r[:, 35])*5 + zscore(r[:, 36])*18)/ (4+5+18) #VAChT

#save receptor density maps 
with open(os.path.join(output_dir, 
                       f'receptor_density_{MASK_NAME}.pickle'), 'wb') as f:
        pickle.dump(receptor_data, f)


#### plotting   
plot_path = os.path.join(output_dir, 'figures', MASK_NAME) 
if not os.path.exists(plot_path):
        os.makedirs(plot_path) 

if MASK_NAME == 'cortical':

    for indx,receptor in enumerate(receptor_names):

        data = masker.inverse_transform(receptor_data[:, indx])
        plotting.plot_img_on_surf(data, surf_mesh='fsaverage',  mask_img=mask_img,
                                        hemispheres=['left', 'right'], views=['lateral', 'medial'],
                                        title=receptor, colorbar=True, cmap = 'plasma')
        fig_fname = 'surface_receptor_'+receptor+'_cortical.png'
        plt.savefig(os.path.join(plot_path, fig_fname))


#### correlation matrix
serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
acetylcholine = ["A4B2", "M1", "VAChT"]
noradrenaline = ["NET"]
opioid = ["MOR"]
glutamate = ["mGluR5"]
histamine = ["H3"]
gaba = ["GABAa"]
dopamine = ["D1", "D2", "DAT"]
cannabinnoid = ["CB1"]
receptor_groups = [serotonin, acetylcholine, noradrenaline, opioid, glutamate, histamine, gaba, dopamine, cannabinnoid]

receptor_data = np.load(os.path.join(home_dir[DATA_ACCESS],'receptors', f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True)
ordered_receptors = [receptor for group in receptor_groups for receptor in group]
df = pd.DataFrame(zscore(receptor_data), columns=receptor_names)
df = df[ordered_receptors]
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title(f'{MASK_NAME}: Correlation of Receptors')
current_pos = 0
for group in receptor_groups:
    group_size = len(group)
    plt.gca().add_patch(Rectangle((current_pos, current_pos), group_size, group_size, fill=False, edgecolor='black', lw=2))
    current_pos += group_size
fig_fname = f'receptor_corr_matrix_{MASK_NAME}.png'
plt.savefig(os.path.join(plot_path, fig_fname))

