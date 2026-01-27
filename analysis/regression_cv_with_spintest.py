"""
This script runs the cross-validated regressions for the receptor/transporter and effect map relationship. 
If specified, it runs the same cross/validation on a null model ('spin test') 
"""

import os
import sys
import glob
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from pathlib import Path
from scipy.stats import zscore
import concurrent.futures
from sklearn.linear_model import LinearRegression
from neuromaps import nulls
from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pickle
import pandas as pd
from scipy.stats import pearsonr
import utils.main_funcs as mf
from config.loader import load_config


def prepare_spins(paths, rec, n_spins=1000):
    """
    Compute spatial spin permutations for surface-based null models.

    Parameters
    ----------
    params : Namespace
        Configuration parameters object.
    paths : Namespace
        Paths configuration object.
    rec : str
        Receptor specification identifier.

    Returns
    -------
    spins : ndarray, shape (n_vertices, n_permutations)
        Spin index permutations.
    """
    receptor_array = mf.load_receptor_array(paths, rec, on_surface=True)
    index_array = np.arange(receptor_array.shape[0])
    spins = nulls.alexander_bloch(
        index_array,
        atlas='fsaverage',
        density='41k',
        n_perm=n_spins
    )
    return spins


def compute_loocv_r2(
    receptor_map_full,
    fmri_activity,
    score,
    model_type,
    receptor_names=None
):
    """
    Perform leave-one-subject-out cross-validation on surface data.

    Parameters
    ----------
    receptor_map_full : ndarray, shape (n_vertices, n_features)
        Surface receptor map.
    fmri_activity : list of ndarray
        List of subject-specific surface activity maps (n_vertices,).
    score : {'determination', 'corr'}, optional
        Scoring metric to use.
    model_type : {'linear', 'lin+quad', 'lin+interact', 'poly2'}, optional
        Regression model specification.
    receptor_names : list of str, optional
        Feature names corresponding to receptors.

    Returns
    -------
    r2_scores : list of float
        Cross-validated R² (or corr²) scores per fold.
    """
    n_subj = len(fmri_activity)
    r2_scores = []

    subj_masks = [
        ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
        for arr in fmri_activity
    ]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    dummy = np.zeros((1, receptor_map_full.shape[1]))
    poly.fit_transform(dummy)
    feature_names = poly.get_feature_names_out(receptor_names)

    if model_type == 'lin+quad':
        mask = [(" " not in n) or ("^" in n) for n in feature_names]
    elif model_type == 'lin+interact':
        mask = ["^" not in n for n in feature_names]
    else:
        mask = np.ones(len(feature_names), dtype=bool)

    for i in range(n_subj):
        X_train_list, y_train_list = [], []
        for j in range(n_subj):
            if j == i:
                continue
            mask_j = subj_masks[j]
            Xj = receptor_map_full[mask_j, :]
            yj = fmri_activity[j].flatten()[mask_j]
            X_train_list.append(zscore(Xj))
            y_train_list.append(zscore(yj))

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        mask_i = subj_masks[i]
        X_test = zscore(receptor_map_full[mask_i, :])
        y_test = zscore(fmri_activity[i].flatten()[mask_i])

        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'poly2':
            model = make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                LinearRegression()
            )
        else:
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            X_train = X_train_poly[:, mask]
            X_test = X_test_poly[:, mask]
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == 'determination':
            r2_scores.append(r2_score(y_test, y_pred))
        elif score == 'corr':
            r, _ = pearsonr(y_test, y_pred)
            r2_scores.append(r ** 2)
        else:
            raise ValueError("score must be 'determination' or 'corr'")

    return r2_scores


def process_task(task, params, paths, rec, spins, output_dir, score, model_type, run_spin):
    """
    Run empirical and spin-based regression CV for a single task.

    Parameters
    ----------
    task : str
        Task identifier.
    params : Namespace
        Configuration parameters object.
    paths : Namespace
        Paths configuration object.
    spins : ndarray
        Precomputed spin permutations.

    Returns
    -------
    results_emp : dict
        Empirical mean CV scores keyed by (task, latent_var).
    results_null : dict
        Null (spin-based) mean CV scores keyed by (task, latent_var).
    """
    results_emp = {}
    results_null = {}

    for latent_var in params.latent_vars:
        if "beta_dir" not in paths:
            beta_dir, add_info = mf.get_beta_dir_and_info(task, params, paths)
        else:
            beta_dir = paths.beta_dir
            add_info = ""

        files = glob.glob(os.path.join(beta_dir ,f"sub-*_{latent_var}_effect_size_map{add_info}.nii.gz"))
        subjects = sorted({
            Path(f).name.split("_")[0].replace("sub-", "")
            for f in files
        })
        subjects = [int(s) for s in subjects]
        subjects = [s for s in subjects if s not in params.ignore]

        fmri_activity = mf.load_surface_effect_maps_for_cv(
            subjects, task, latent_var, beta_dir, add_info
        )
        receptor_array = mf.load_receptor_array(paths, rec, on_surface=True)

        print(f"--- CV for {task} and {latent_var} ---")
        r2_scores = compute_loocv_r2(
            receptor_array, fmri_activity, score=score, model_type=model_type
        )

        with open(
            os.path.join(
                output_dir, 
                f'{task}_{latent_var}_all_regression_cv_r2_{model_type}_{score}.pickle'
            ),
            "wb"
        ) as fp:
            pickle.dump(r2_scores, fp)

        results_emp[(task, latent_var)] = np.nanmean(r2_scores)

        if run_spin and model_type == 'linear':
            print(f"--- Spin test for {task} and {latent_var} ---")
            all_null = []
            for s in range(spins.shape[1]):
                print(f"--- spin {s} out of {spins.shape[1]} ---")
                receptor_spin = receptor_array[spins[:, s], :]
                r2_scores = compute_loocv_r2(
                    receptor_spin, fmri_activity, score=score, model_type=model_type
                )
                all_null.append(r2_scores)

            with open(
                os.path.join(
                    output_dir, 
                    f'{task}_{latent_var}_all_regression_null_cv_r2_{score}.pickle'
                ),
                "wb"
            ) as fp:
                pickle.dump(all_null, fp)

            results_null[(task, latent_var)] = np.nanmean(
                [np.nanmean(r2) for r2 in all_null]
            )

        elif run_spin and model_type != 'linear':
            raise NotImplementedError(
                "Spin test not implemented for non-linear models."
            )

    return results_emp, results_null


def run_reg_cv_with_spin(params, paths, rec, spins=None, output_dir='', score='determination', model_type = 'linear', run_spin=True):
    """
    Run regression CV (and optional spin tests) across all tasks.

    Parameters
    ----------
    params : Configuration parameters object.
    paths : Paths configuration object.
    spins : Precomputed spin permutations.
    output_dir : Directory to store outputs.
    score : {'determination', 'corr'}, optional
        Scoring metric.
    """
    all_emp, all_null = {}, {}

    os.makedirs(output_dir, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_task, task, params, paths, rec, spins, output_dir, score, model_type, run_spin)
            for task in params.tasks
        ]
        for f in futures:
            emp, null = f.result()
            all_emp.update(emp)
            all_null.update(null)

    if len(params.tasks) > 1:
        df_emp = pd.DataFrame.from_dict(all_emp, orient="index", columns=["r2"])
        df_emp.index = pd.MultiIndex.from_tuples(
            df_emp.index, names=["task", "latent_var"]
        )
        df_emp.to_csv(
            os.path.join(output_dir, f"overview_regression_cv_{score}.csv")
        )

        if run_spin:
            df_null = pd.DataFrame.from_dict(all_null, orient="index", columns=["r2"])
            df_null.index = pd.MultiIndex.from_tuples(
                df_null.index, names=["task", "latent_var"]
            )
            df_null.to_csv(
                os.path.join(output_dir, f"overview_null_cv_{score}.csv")
            )


# Main
if __name__ == "__main__":
    N_SPINS = 1000
    MODEL_TYPE = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'
    RUN_SPIN = True
    SCORE = 'determination' # determination or corr

    params, paths, rec = load_config('all', return_what='all')

    output_dir = os.path.join(paths.home_dir, 'variance_explained')
    os.makedirs(output_dir, exist_ok=True)

    spins = prepare_spins(paths, rec, n_spins=N_SPINS)

    run_reg_cv_with_spin(params, paths, spins, output_dir, score=SCORE, model_type=MODEL_TYPE, run_spin = RUN_SPIN)

