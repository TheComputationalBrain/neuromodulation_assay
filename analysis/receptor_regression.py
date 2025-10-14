#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:25:09 2024

@author: Alice Hodapp

This gets the receptor density data and the output of the GLM to run a multiple regression and/or dominance analyis seperatly for each subejct and 
surprise, confidence, predictability. The output corresponds to the regression coefficients and the total dominance, respectivly.
"""

import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import main_funcs as mf
from params_and_paths_analysis import Paths, Params, Receptors
from dominance_funcs import dominance_stats

MODEL_TYPE = 'linear'  # 'linear','lin+interact'
RUN_REGRESSION = True
RUN_DOMINANCE = True 
NUM_WORKERS = 30  # Set an appropriate number of workers to run dominance code in parallel
START_AT = 0 #In case the dominance analysis was or had to be interrupted at some point, put the last processed subject here 

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

if RUN_REGRESSION:
    for task in params.tasks:  

        beta_dir,_ = mf.get_beta_dir_and_info

        output_dir = os.path.join(beta_dir, 'regressions', rec.source) #! different way to egt name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 

        receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 

        receptor_density = mf.load_recptor_array(on_surface = False)

        if MODEL_TYPE == 'linear':
            columns = rec.receptor_names + ["R2", "adjusted_R2", "BIC"]
        else:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            dummy = np.zeros((1, receptor_density.shape[1]))
            poly_features = poly.fit_transform(dummy)
            feature_names = poly.get_feature_names_out(rec.receptor_names)
            if MODEL_TYPE == 'lin+quad':
            # Linear + quadratic only (exclude interaction terms)
                mask = [
                    (" " not in name) or ("^" in name)  # keep original features and squared terms
                    for name in feature_names
                ]
                feature_names = feature_names[mask]
            elif MODEL_TYPE == 'lin+interact':
                # Linear + interactions only (exclude squared terms)
                mask = [
                    "^" not in name  # keep everything that is NOT quadratic
                    for name in feature_names
                                    ]
                feature_names = feature_names[mask]
            columns = list(feature_names) + ["R2", "adjusted_R2", "BIC"]

        for latent_var in params.latent_vars:
            results_df = pd.DataFrame(columns=columns)

            for sub in subjects: 
                
                y_data = mf.load_effect_map_array(sub, task, latent_var)
                receptor_density_zm = receptor_density

                non_nan_indices = ~np.isnan(y_data)
                X = receptor_density_zm[non_nan_indices,:] #non parcelated data might contain a few NaNs from voxels with constant activation 
                y = y_data[non_nan_indices]

                if MODEL_TYPE == 'linear':
                    # Linear regression only
                    model = LinearRegression()

                elif MODEL_TYPE == 'poly2':
                    # Full second-degree polynomial (linear + quadratic + interactions)
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    model = make_pipeline(poly, LinearRegression())

                else:
                    # Fit polynomial features to training data only
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X = poly.fit_transform(X)

                    feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

                    if MODEL_TYPE == 'lin+quad':
                        # Linear + quadratic only (exclude interaction terms)
                        mask = [
                            (" " not in name) or ("^" in name)  # keep original features and squared terms
                            for name in feature_names
                        ]
                        # Apply mask to filter features
                    elif MODEL_TYPE == 'lin+interact':
                        # Linear + interactions only (exclude squared terms)
                        mask = [
                            "^" not in name  # keep everything that is NOT quadratic
                            for name in feature_names
                        ]
                        # Apply mask to filter features
                    X = X[:, mask]
                    filtered_feature_names = feature_names[mask]

                    # Fit model
                    model = LinearRegression()
            

                model.fit(X, y)
                if MODEL_TYPE == 'poly2':
                    coefs = model.named_steps['linearregression'].coef_
                else:
                    coefs = model.coef_
                yhat = model.predict(X)
                SS_Residual = sum((y - yhat) ** 2)
                SS_Total = sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (float(SS_Residual)) / SS_Total
                adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1) 

            results = pd.DataFrame([np.append(coefs, [r_squared, adjusted_r_squared])], columns = columns)
            results_df = pd.concat([results_df,results], ignore_index=True)

        fname = f'{latent_var}_{params.mask}_regression_results_bysubject_all_{MODEL_TYPE}.csv'
        results_df.to_csv(os.path.join(output_dir, fname), index=False)  

if RUN_DOMINANCE:
    def process_subject(sub, latent_var, task):
        print(f"--- dominance analysis for {task} subject {sub} ----")

        beta_dir,_ = mf.get_beta_dir_and_info

        output_dir = os.path.join(beta_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True) 

        y_data = mf.load_effect_map_array(sub, task, latent_var)

        non_nan_indices = ~np.isnan(y_data)
        X = receptor_density[non_nan_indices,:] #non parcelated data might contain a few NaNs from voxels with constant activation 
        y = y_data[non_nan_indices]
        if MODEL_TYPE == 'lin+quad':
            # Fit polynomial features to training data only
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X = poly.fit_transform(X)

            feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

            if MODEL_TYPE == 'lin+quad':
                # Linear + quadratic only (exclude interaction terms)
                mask = [
                    (" " not in name) or ("^" in name)  # keep original features and squared terms
                    for name in feature_names
                ]
            X = X[:, mask]
            filtered_feature_names = feature_names[mask]
            m = dominance_stats(X, y, feature_names=filtered_feature_names)
        elif MODEL_TYPE == 'linear':
            m = dominance_stats(X, y)
        else:
            raise ValueError(f"Dominance analysis for '{MODEL_TYPE}' not possible!")

        with open(os.path.join(output_dir, f'{latent_var}_{params.mask}_dominance_sub-{sub:02d}_{MODEL_TYPE}.pickle'), 'wb') as f:
            pickle.dump(m, f)
        total_dominance_array = m["total_dominance"]
        results = pd.DataFrame([total_dominance_array], columns=rec.receptor_names)
        return results

    for task in params.tasks:
        for latent_var in params.variables:
            print(f"--- dominance analysis for {latent_var} ----")
            results_df = pd.DataFrame(columns=rec.receptor_names)
            subjects = [sub for sub in subjects if sub > START_AT]
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(process_subject, sub, latent_var, task) for sub in subjects]
                for future in futures:
                    results = future.result()
                    results_df = pd.concat([results_df, results], ignore_index=True)

        # Save data
        results_df.to_pickle(os.path.join(output_dir, f'{latent_var}_{params.mask}_dominance_allsubj_{MODEL_TYPE}.pickle'))
