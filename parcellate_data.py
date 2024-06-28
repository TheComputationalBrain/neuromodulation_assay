import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
import pickle 
import os
import matplotlib.pyplot as plt
import main_funcs as mf
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.input_data import NiftiLabelsMasker
from params_and_paths import *


RUN_OLS = False #TODO: set this as global flag

contrasts = ['surprise','confidence', 'predictability', 'predictions']
atlas = fetch_atlas_schaefer_2018(n_rois=int(mask_details[MASK_NAME]), resolution_mm=2) 
atlas.labels = np.insert(atlas.labels, 0, "Background")
masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate

fmri_dir = mf.get_fmri_dir(DB_NAME)
subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

if RUN_OLS:
    output_dir = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level','OLS')
else:
    output_dir = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level')

for sub in subjects:
    for contrast_id in contrasts:
        fname = f'sub-{sub:02d}_{contrast_id}_{MASK_NAME}_effect_size_map.nii.gz'
        effect_size = nib.load(os.path.join(output_dir,fname))
        effects_parcel = masker.fit_transform(effect_size)
        with open(os.path.join(output_dir, 
                            f'sub-{sub:02d}_{contrast_id}_{MASK_NAME}_{mask_details[MASK_NAME]}_effect_size.pickle'), 'wb') as f:
            pickle.dump(effects_parcel, f)