
import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import zscore
import concurrent.futures
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from neuromaps import nulls, transforms
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import main_funcs as mf
import glob
import pickle
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

paths = Paths()
params = Params()
rec = Receptors()

n_spins = 1000
model_type = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'
EXPLORE_MODEL = 'noEntropy_noER'

suffix = ''

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
latent_vars = ['surprise', 'confidence'] 

output_dir = os.path.join(paths.home_dir, 'variance_explained')
os.makedirs(output_dir, exist_ok=True)

fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data'),
            'lanA': '/home_local/alice_hodapp/language_localizer/'}

ignore = {'NAConf': [3, 5, 6, 9, 36, 51],
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': [],
            'lanA' : []}

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)
receptor_data = zscore(np.load(os.path.join(receptor_dir, f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
index_array = np.arange(receptor_data.shape[0])
spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k', n_perm=n_spins)

def process_task(task):
    if task == 'Explore':
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
    elif task == 'lanA':
        beta_dir = fmri_dir['lanA']
    else:
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')    
        
    subjects = [subj for subj in mf.get_subjects(task, fmri_dir[task]) if subj not in ignore[task]]

    add_info = '_firstTrialsRemoved' if task == 'NAConf' else ''
    
    for latent_var in latent_vars:
        print(f"--- Spin test for {task} and {latent_var} ---")
        if task == 'lanA':
            fmri_files = []
            for subj in subjects:
                subj_id = f"{subj:03d}"  
                pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
                fmri_files.extend(glob.glob(pattern))
        else:
            fmri_files = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz')))        

        fmri_activity = []
        all_null = []
        for file in fmri_files:
            data_vol = nib.load(file)
            effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
            data_gii = [img.agg_data().T for img in effect_data]
            fmri_activity.append(np.hstack(data_gii))

        for s in range(spins.shape[1]):
            receptor_spin = receptor_data[spins[:, s], :]
            r2_scores = []
            
            skipped_subjects = []

            for i, subj_id in enumerate(subjects):
                X_train_blocks = []
                y_train_blocks = []

                for j in range(len(subjects)):
                    if j == i:
                        continue
                    mask_j = ~np.logical_or(np.isnan(fmri_activity[j]), np.isclose(fmri_activity[j], 0)).flatten()
                    valid_data = fmri_activity[j].flatten()[mask_j]

                    if valid_data.size == 0: #safty check
                        print(f"Skipping subject {subjects[j]} in training (no valid voxels)")
                        skipped_subjects.append(subjects[j])
                        continue

                    X_train_blocks.append(receptor_spin[mask_j])
                    y_train_blocks.append(valid_data)

                if not X_train_blocks or not y_train_blocks: #safty check
                    print(f"Skipping fold for test subject {subj_id} (no training data left)")
                    skipped_subjects.append(subj_id)
                    continue

                # Concatenate training
                X_train = np.concatenate(X_train_blocks, axis=0)
                y_train = np.concatenate(y_train_blocks, axis=0)

                # Scaling
                X_scaler = StandardScaler().fit(X_train)
                X_train_scaled = X_scaler.transform(X_train)
                y_mean = np.nanmean(y_train)
                y_std = np.nanstd(y_train)
                y_train_scaled = (y_train - y_mean) / y_std

                if y_std == 0:
                    print(f"⚠️ y_train has zero variance for fold {i}, skipping...")
                    continue

                # Prepare test
                mask_i = ~np.logical_or(np.isnan(fmri_activity[i]), np.isclose(fmri_activity[i], 0)).flatten()
                valid_test = fmri_activity[i].flatten()[mask_i]

                if valid_test.size == 0: #safty check
                    print(f"Skipping test subject {subj_id} (no valid voxels)")
                    skipped_subjects.append(subj_id)
                    continue

                X_test = receptor_spin[mask_i]
                y_test_scaled = (valid_test - y_mean) / y_std
                X_test_scaled = X_scaler.transform(X_test)

                if model_type == 'linear':
                    # Linear regression only
                    model = LinearRegression()

                elif model_type == 'poly2':
                    # Full second-degree polynomial (linear + quadratic + interactions)
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    model = make_pipeline(poly, LinearRegression())

                else:
                    # Fit polynomial features to training data only
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)

                    feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

                    if model_type == 'lin+quad':
                        # Linear + quadratic only (exclude interaction terms)
                        mask = [
                            (" " not in name) or ("^" in name)  # keep original features and squared terms
                            for name in feature_names
                        ]

                    if model_type == 'lin+interact':
                        # Linear + interactions only (exclude squared terms)
                        mask = [
                            "^" not in name  # keep everything that is NOT quadratic
                            for name in feature_names
                        ]

                    # Apply mask to filter features
                    X_train = X_train_poly[:, mask]
                    X_test = X_test_poly[:, mask]
                    filtered_feature_names = feature_names[mask]

                    print(filtered_feature_names)

                    model = LinearRegression()

                #model fit and predict left out subject
                model.fit(X_train_scaled, y_train_scaled)
                y_pred_scaled = model.predict(X_test_scaled)
                r2 = r2_score(y_test_scaled, y_pred_scaled)
                all_null.append(r2)

            if skipped_subjects:
                print(f"\nSummary: skipped {len(set(skipped_subjects))} subjects: {sorted(set(skipped_subjects))}\n")
        
        if model_type == 'linear':
            with open(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2{suffix}.pickle'), "wb") as fp:
                pickle.dump(all_null, fp)
        else:
            with open(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2_{model_type}{suffix}.pickle'), "wb") as fp:
                pickle.dump(all_null, fp)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_task, tasks)

    


