#normalize single subject paramter estimates in native space 
import glob
import sys
import os
import os.path as op
import nibabel as nib
import numpy as np
from pypreprocess.pypreprocess.subject_data import SubjectData # for SPM path
from nipype.interfaces.spm import Normalize12 

SUBJECT = 2
estimate_native = op.join('/neurospin/unicog/protocols/IRMf/EncodeProb_BounmyMeyniel_2020/derivatives/first_level_estimates/sub-{SUBJECT:02d}/model_surp_confidence', 'sub-{SUBJECT:02d}_confidence_pos_effect_size_native_smNone.nii.gz') 

# Locate the deformation file
deformation_file = op.join('/neurospin/unicog/protocols/IRMf/EncodeProb_BounmyMeyniel_2020/bids_dataset/sub-{SUBJECT:02d}/anat', f'y_sub-{SUBJECT:02d}_T1w.nii')

# Write normalized functional files
func_voxel_size = [1.5, 1.5, 1.5] # in mm

norm12 = Normalize12()
norm12.inputs.jobtype = 'write'
norm12.inputs.affine_regularization_type = 'mni'
norm12.inputs.deformation_file = deformation_file
norm12.inputs.apply_to_files = estimate_native
norm12.inputs.write_voxel_sizes = func_voxel_size
norm12.inputs.out_prefix = 'mni_'
norm12.run()
print(f'Normalization of functional files for subject {SUBJECT} done.')