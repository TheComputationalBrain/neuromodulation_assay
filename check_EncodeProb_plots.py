#this script normalizes the native space effect maps and plots the group leel effects to compare them to my results 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template, fetch_surf_fsaverage
from nilearn import surface
from sklearn.linear_model import LinearRegression
import main_funcs as mf
import fmri_funcs as fun
import nibabel as nib


data_path = '/home_local/tiffany_bounmy/EncodeProb_BounmyMeyniel_2020/derivatives/first_level_estimates'
second_level_dir = '/home_local/alice_hodapp/NeuroModAssay/EncodeProb/native_to_mni/second_level_estimates'

subjects = [2,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30]

template = load_mni152_template(resolution=2)

#group level analyis
eff_size_files = []
for sub in subjects: 
    #get contrast data (beta estimates) and concatinate
    ef_size = nib.load(os.path.join(data_path,f'sub-{sub:02d}','model_surp_confidence',f'sub-{sub:02d}_confidence_pos_effect_size_mni_smNone.nii.gz'))
    eff_size_files.append(ef_size)

#second level model:
design_matrix = pd.DataFrame([1] * len(eff_size_files),
                                columns=['intercept']) #one sample test
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(eff_size_files,
                                            design_matrix=design_matrix)
z_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='z_score')

fsaverage = fetch_surf_fsaverage()
texture = surface.vol_to_surf(z_map, fsaverage.pial_right)
plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture, hemi='right',
    colorbar=True,
    bg_map=fsaverage.sulc_right)
fname = f'confidence_pos_z_map_right.png' 
plt.savefig(os.path.join(second_level_dir, fname))
