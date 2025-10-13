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
import nibabel as nib
from math import log
from neuromaps import transforms
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import zscore
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors
from dominance_stats import dominance_stats

MODEL_TYPE = 'lin+quad'  # 'linear', 'poly2', 'lin+quad', 'lin+interact'
RUN_REGRESSION = True
RUN_DOMINANCE = True 
ON_SURFACE = False
NUM_WORKERS = 30  # Set an appropriate number of workers to run dominance code in parallel
START_AT = 0 #In case the dominance analysis was or had to be interrupted at some point, put the last processed subject here 

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

if ON_SURFACE:
    proj = '_surf'
else:
    proj = ''

if params.update:
    if params.db == 'Explore':
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model', params.model)
    else:
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'update_model')
else:
    if params.db == 'Explore':
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level',params.model)
    else:
        beta_dir = os.path.join(paths.home_dir,params.db,params.mask,'first_level')
                                    
if params.parcelated:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
    output_dir = os.path.join(beta_dir, 'regressions', rec.source)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    mask_comb = params.mask + '_' + params.mask_details 
    if rec.source !='AHBA':
        receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
    else:
        gene_expression = pd.read_csv(os.path.join(receptor_dir,f'gene_expression_complex_desikan.csv'))
        receptor_density = zscore(gene_expression.to_numpy(), nan_policy='omit')
else:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) #voxel level analyis can only be run on PET data densities 
    output_dir = os.path.join(beta_dir, 'regressions', rec.source)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}{proj}.pickle'), allow_pickle=True), nan_policy='omit')
    mask_comb = params.mask 

if rec.source == 'autorad_zilles44':
    #autoradiography dataset is only one hemisphere 
    receptor_density = np.concatenate((receptor_density, receptor_density))

if params.db in ['NAConf']:
    add_info = '_firstTrialsRemoved'
elif not params.zscore_per_session:
    add_info = '_zscoreAll'
else:
    add_info = ""

def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic


print(f'------- running regressions with {rec.source} as receptor density -------')

if RUN_REGRESSION:
    #sklearn regression
    if params.db == 'Explore':
        variables = ['confidence', 'surprise']
    else:
        variables = params.latent_vars
        
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

    for latent_var in variables:
        results_df = pd.DataFrame(columns=columns)

        for sub in subjects: 
            
            if ON_SURFACE:
                data_vol = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size_map{add_info}.nii.gz'))
                effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k', method='nearest')
                data_gii = []
                for img in effect_data:
                    data_hemi = img.agg_data()
                    data_hemi = np.asarray(data_hemi).T
                    data_gii += [data_hemi]
                    y_data = np.hstack(data_gii)    
                zeromask = np.isclose(y_data, 0)
                mask = np.logical_not(zeromask)
                y_data, receptor_density_zm = y_data[mask], receptor_density[mask,:]
            else: 
                y_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()
                receptor_density_zm = receptor_density

            if params.parcelated:
                non_nan_region = ~np.isnan(receptor_density_zm).any(axis=1)
                non_nan_indices = np.where(non_nan_region)[0]
                X = receptor_density_zm[non_nan_indices,:] #manual assignment of autored data means that some regions are empty
                y = y_data[non_nan_indices]
            else:
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
                num_params = len(coefs) + 1
                mse = mean_squared_error(y, yhat)
                bic = calculate_bic(len(y), mse, num_params)

            #results by functional activity across participants 
            results = pd.DataFrame([np.append(coefs, [r_squared, adjusted_r_squared, bic])], columns = columns)
            results_df = pd.concat([results_df,results], ignore_index=True)

        if MODEL_TYPE == 'linear':
            fname = f'{latent_var}_{mask_comb}_regression_results_bysubject_all{proj}.csv'
        else:
            fname = f'{latent_var}_{mask_comb}_regression_results_bysubject_all{proj}_{MODEL_TYPE}.csv'
        results_df.to_csv(os.path.join(output_dir, fname), index=False)  

if RUN_DOMINANCE:
    variables = ['confidence', 'surprise']

    def process_subject(sub, latent_var):
        print(f"--- dominance analysis for subject {sub} ----")
        y_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()
        if params.parcelated:
            non_nan_region = ~np.isnan(receptor_density).any(axis=1)
            non_nan_indices = np.where(non_nan_region)[0]
            X = receptor_density[non_nan_indices,:] #manual assignment of autored data means that some regions are empty
            y = y_data[non_nan_indices]
        else:
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

    for latent_var in variables:
        print(f"--- dominance analysis for {latent_var} ----")
        results_df = pd.DataFrame(columns=rec.receptor_names)
        subjects = [sub for sub in subjects if sub > START_AT]
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_subject, sub, latent_var) for sub in subjects]
            for future in futures:
                results = future.result()
                results_df = pd.concat([results_df, results], ignore_index=True)

        # Save data
        if MODEL_TYPE == 'linear':
            results_df.to_pickle(os.path.join(output_dir, f'{latent_var}_{mask_comb}_dominance_allsubj.pickle'))
        else:
            results_df.to_pickle(os.path.join(output_dir, f'{latent_var}_{mask_comb}_dominance_allsubj_{MODEL_TYPE}.pickle'))
