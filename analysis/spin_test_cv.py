
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
from params_and_paths_analysis import Paths, Params, Receptors
from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import main_funcs as mf
import pickle
import pandas as pd
from scipy.stats import pearsonr


paths = Paths()
params = Params()
rec = Receptors()

N_SPINS = 1000
MODEL_TYPE = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'
RUN_SPIN = False
EXPLORE_MODEL = 'noEntropy_noER'
FIX_PARTIAL_COVERAGE = True
SCORE = 'determination' # determination or corr


output_dir = os.path.join(paths.home_dir, 'variance_explained')
os.makedirs(output_dir, exist_ok=True)

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_data =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
index_array = np.arange(receptor_data.shape[0])
spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k', n_perm=N_SPINS)

def compute_loocv_r2_by_subject_mask(receptor_map_full, fmri_activity, score='determination'):
    """
    receptor_map_full: (n_vertices, n_features)
    fmri_activity: list of (n_vertices,) arrays for each subject
    Returns: list of per-fold R² or corr² scores.
    """
    n_subj = len(fmri_activity)
    r2_scores = []

    # Precompute masks for each subject
    subj_masks = [
        ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
        for arr in fmri_activity
    ]

    for i in range(n_subj):
        # --- Training data ---
        X_train_list, y_train_list = [], []
        for j in range(n_subj):
            if j == i: 
                continue

            mask_j = subj_masks[j]
            Xj = receptor_map_full[mask_j, :]
            yj = fmri_activity[j].flatten()[mask_j]

            # Normalize within-subject
            yj = (yj - np.nanmean(yj)) / (np.nanstd(yj) + 1e-12)

            X_train_list.append(Xj)
            y_train_list.append(yj)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # --- Test data (held-out subject) ---
        mask_i = subj_masks[i]
        X_test = receptor_map_full[mask_i, :]
        y_test = fmri_activity[i].flatten()[mask_i]
        y_test = (y_test - np.nanmean(y_test)) / (np.nanstd(y_test) + 1e-12)

        # --- Model fitting ---
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == 'determination':
            r2_scores.append(r2_score(y_test, y_pred))
        elif score == 'corr':
            r, _ = pearsonr(y_test, y_pred)
            r2_scores.append(r**2)
        else:
            raise ValueError("score must be 'determination' or 'corr'")

    return r2_scores


def process_task(task):
    results_emp = {}
    results_null = {}
    
    for latent_var in params.latent_vars:
        subject_paths = params.home_dir if task == "lanA" else params.root_dir
        subjects = mf.get_subjects(task, os.path.join(subject_paths, paths.fmri_dir[task]))
        subjects = [subj for subj in subjects if subj not in params.ignore[task]] 

        fmri_activity = mf.load_surface_effect_maps_for_cv(subjects, task, latent_var)

        print(f"--- CV for {task} and {latent_var} ---")  
        r2_scores = compute_loocv_r2_by_subject_mask(receptor_data, fmri_activity, score=SCORE)
        with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r_2_{SCORE}.pickle'), "wb") as fp:   
            pickle.dump(r2_scores, fp)
        average_r2 = np.mean(r2_scores)
        results_emp[(task, latent_var)] = average_r2
        
        if RUN_SPIN and MODEL_TYPE == 'linear':
            print(f"--- Spin test for {task} and {latent_var} ---")
            all_null = []
            for s in range(spins.shape[1]):
                receptor_spin = receptor_data[spins[:, s], :]
                r2_scores = compute_loocv_r2_by_subject_mask(receptor_spin, fmri_activity, score=SCORE)
                all_null.append(r2_scores)
            with open(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2_{SCORE}.pickle'), "wb") as fp:
                pickle.dump(all_null, fp)
            all_null_means = [np.mean(r2) for r2 in all_null]
            average_r2 = np.mean(all_null_means)
            results_null[(task, latent_var)] = average_r2
        elif RUN_SPIN and MODEL_TYPE != 'linear':
            raise NotImplementedError("Spin test not implemented for non-linear models.")
        else:
            pass

    return results_emp, results_null


# Main
all_emp, all_null = {}, {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_task, task) for task in params.tasks]
    for f in futures:
        emp, null = f.result()
        all_emp.update(emp)
        all_null.update(null)

# Convert to DataFrames
df_emp = pd.DataFrame.from_dict(all_emp, orient="index", columns=["r2"])
df_emp.index = pd.MultiIndex.from_tuples(df_emp.index, names=["task", "latent_var"])
df_emp.to_csv(os.path.join(output_dir, f"overview_regression_cv_{SCORE}.csv"))

df_null = pd.DataFrame.from_dict(all_null, orient="index", columns=["r2"])
df_null.index = pd.MultiIndex.from_tuples(df_null.index, names=["task", "latent_var"])
df_null.to_csv(os.path.join(output_dir, f"overview_null_cv_{SCORE}.csv"))

