# compare the correlation between each of the group level map comparisons to a null model of 1000 random spins
#this supplements the overlap plots across studies for each of the latent varibales in fig. 3 of the manuscript

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.surface import SurfaceImage
from params_and_paths import Paths, Params

#load debugged local version --> see pull request on neuromaps github
sys.path.insert(0, os.path.abspath("."))
from neuromaps import nulls, transforms
from neuromaps import stats

paths = Paths()
params = Params()

#loop over surpise and confidence
tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
contrasts = ['surprise', 'confidence']

CORR_WITHIN = False  #False=get correlations between confidence and suprise maps instead (seperated)
EXPLORE_MODEL = 'noEntropy_noER'
spin_results = []

output_dir = os.path.join(paths.home_dir, 'domain_general')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

import itertools

spin_results = []

if CORR_WITHIN:
    for contrast in contrasts:
        #process all maps and null models for each dataset
        surf_maps = {}
        null_models = {}

        for task in tasks:
            add_info = '_firstTrialsRemoved' if task == 'NAConf' else ""

            if task == 'Explore':
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
            else:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
            
            img_path = os.path.join(group_dir, f'{contrast}_schaefer_effect_map{add_info}.nii.gz')
            img = nib.load(img_path)

            # surface projection
            surf_data = transforms.mni152_to_fsaverage(img, fsavg_density='41k')
            data_gii = [img.agg_data().T for img in surf_data]
            surf_map = np.hstack(data_gii)
            surf_maps[task] = surf_map

            # null model for each map
            surf_map_list = [surf_map]
            rotated = nulls.alexander_bloch(
                surf_map_list, atlas='fsaverage', density='41k',
                n_perm=1000, seed=1234
            )
            null_models[task] = rotated

        for task1, task2 in itertools.combinations(tasks, 2):
            print(f"--- Spin test for {contrast}: {task1} and {task2} ---")
            surf_map1 = [surf_maps[task1]]
            surf_map2 = [surf_maps[task2]]
            rotated = null_models[task1]  # use task1's null model

            corr, pval = stats.compare_images(surf_map1, surf_map2, nulls=rotated)

            spin_results.append({
                'task1': task1,
                'task2': task2,
                'contrast': contrast,
                'corr': corr,
                'pval': pval
            })

    spin_results = pd.DataFrame(spin_results)
    spin_results.to_csv(os.path.join(output_dir, 'results_map_spin.csv'), index = False)
else:
    surf_maps = {}
    null_models = {}

    for task in tasks:
        add_info = '_firstTrialsRemoved' if task == 'NAConf' else ""

        for contrast in contrasts:
            if task == 'Explore':
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
            else:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')

            img_path = os.path.join(group_dir, f'{contrast}_schaefer_effect_map{add_info}.nii.gz')
            img = nib.load(img_path)

            # Surface projection
            surf_data = transforms.mni152_to_fsaverage(img, fsavg_density='41k')
            data_gii = [img.agg_data().T for img in surf_data]
            surf_map = np.hstack(data_gii)

            # Store map and null model using task+contrast as key
            key = f'{task}_{contrast}'
            surf_maps[key] = surf_map

            # Generate null model for this specific map
            rotated = nulls.alexander_bloch(
                [surf_map], atlas='fsaverage', density='41k',
                n_perm=1000, seed=1234
            )
            null_models[key] = rotated

    # Run all comparisons
    spin_results = []

    for task1, task2 in itertools.product(tasks, repeat=2):
        # 1. confidence(task1) vs surprise(task2)
        map1_key = f'{task1}_confidence'
        map2_key = f'{task2}_surprise'
        surf_map1 = [surf_maps[map1_key]]
        surf_map2 = [surf_maps[map2_key]]
        rotated = null_models[map1_key]  # Use null model of map1
        corr, pval = stats.compare_images(surf_map1, surf_map2, nulls=rotated)

        spin_results.append({
            'task1': task1,
            'contrast1': 'confidence',
            'task2': task2,
            'contrast2': 'surprise',
            'corr': corr,
            'pval': pval
        })

        # 2. surprise(task1) vs confidence(task2)
        map1_key = f'{task1}_surprise'
        map2_key = f'{task2}_confidence'
        surf_map1 = [surf_maps[map1_key]]
        surf_map2 = [surf_maps[map2_key]]
        rotated = null_models[map1_key]
        corr, pval = stats.compare_images(surf_map1, surf_map2, nulls=rotated)

        spin_results.append({
            'task1': task1,
            'contrast1': 'surprise',
            'task2': task2,
            'contrast2': 'confidence',
            'corr': corr,
            'pval': pval
        })

    # Save to CSV
    spin_results_df = pd.DataFrame(spin_results)
    spin_results_df.to_csv(os.path.join(output_dir, 'results_map_spin_across.csv'), index = False)
