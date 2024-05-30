#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:32:01 2024

@author: Alice Hodapp
"""

import numpy as np
import nibabel as nib
from scipy.stats import zscore
import pickle 
import os
from nilearn.input_data import NiftiMasker
from nilearn import surface, datasets
from netneurotools import plotting
from params_and_paths import *

receptor_path = os.path.join(home_dir[DATA_ACCESS],'receptors','PET_nifti_images') #path to downloaded data from Hansen et al. (2020)
output_dir = os.path.join(home_dir[DATA_ACCESS],'receptors')

masker = NiftiMasker(mask_img=MASK)
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

# make final voxel x receptor matrix  
receptor_data = np.zeros([r.shape[0], len(receptor_names)])
receptor_data[:, 0] = zscore(r[:, 0])
receptor_data[:, 2:9] = zscore(r[:, 3:10])
receptor_data[:, 10:14] = zscore(r[:, 12:16])
receptor_data[:, 15:18] = zscore(r[:, 19:22])

# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# weighted average of VAChT FEOBV
#receptor_data[:, 18] = (zscore(r[:, 22])*3 + zscore(r[:, 23])*4 + zscore(r[:, 24])*5 + zscore(r[:, 25])*18) / (3+4+5+18) #Schmitz & Spreng dataset not available in volumetric form yet
receptor_data[:, 18] = (zscore(r[:, 23])*4 + zscore(r[:, 24])*5 + zscore(r[:, 25])*18) / (4+5+18) 

#save receptor density maps 
with open(os.path.join(output_dir, 
                       f'receptor_density_{MASK_NAME}.pickle'), 'wb') as f:
        pickle.dump(receptor_data, f)


#### plotting     
plot_path = os.path.join(output_dir, 'figures', MASK_NAME) 
if not os.path.exists(plot_path):
        os.makedirs(plot_path)

fsaverage = datasets.fetch_surf_fsaverage(mesh = 'fsaverage') #for best comparability with Hansen et al 2022

for k in range(len(receptor_names)):

    data = masker.inverse_transform(receptor_data[:, k])
    fsavg_lh = surface.vol_to_surf(data, fsaverage.pial_left)
    fsavg_rh = surface.vol_to_surf(data, fsaverage.pial_right)

    data_array = np.concatenate((fsavg_lh, fsavg_rh))

    zeromask = data_array == 0 #quick implementation to ignore data that was masked (however voxels that are zero are also ignored)

    #currently using the netneurotools wrapper for best comparibility to paper
    brain = plotting.plot_fsvertex(data=data_array,
                                    colormap='plasma',
                                    views=['lat', 'med'],
                                    mask=zeromask,
                                    colorbar=True,
                                    data_kws={'representation': "wireframe"})
    
    brain.save_image(plot_path+'surface_receptor_'+receptor_names[k]+'.png')