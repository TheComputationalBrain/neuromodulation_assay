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
import picklde
from math import log
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import main_funcs as mf
from params_and_paths import *
from dominance_stats import dominance_stats

RUN_REGRESSION = True
RUN_DOMINANCE = True
FROM_OLS = False
PARCELATED = True
NUM_WORKERS = 26  # Set an appropriate number of workers to run code in parallel


fmri_dir = mf.get_fmri_dir(DB_NAME)
subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

if FROM_OLS:
    beta_dir  = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level', 'OLS')
else: 
    beta_dir  = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level')
                                    
if PARCELATED:
    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors', RECEPTOR_SOURCE)  
    output_dir = os.path.join(beta_dir, 'regressions', RECEPTOR_SOURCE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    mask_comb = MASK_NAME + '_' + mask_details[MASK_NAME] 
    X_data = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True)) 

else:

    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors', 'PET') #vertex level analyis can only be run on PET data densities 
    output_dir = os.path.join(beta_dir, 'regressions', 'PET')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    X_data = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True))
    mask_comb = MASK_NAME 


receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
y_names = np.array(['surprise', 'confidence', 'predictability', 'predictions'])

if RUN_REGRESSION:
    #sklearn regression
    columns = np.concatenate((receptor_names, np.array(["R2", "adjusted_R2", "BIC"])))

    def calculate_bic(n, mse, num_params):
        bic = n * log(mse) + num_params * log(n)
        return bic

    for y_name in y_names:
        results_df = pd.DataFrame(columns=columns)

        for sub in subjects: 
            
            y_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{y_name}_{mask_comb}_effect_size.pickle'), allow_pickle=True).flatten()
            non_nan_indices = ~np.isnan(y_data)
            X = X_data[non_nan_indices,:] #non parcelated data might contain a few NaNs from voxels with constant activation 
            y = y_data[non_nan_indices]

            lin_reg = LinearRegression()
            lin_reg.fit(X, y)
            yhat = lin_reg.predict(X)
            coefs = lin_reg.coef_

            #adjusted R2
            SS_Residual = sum((y - yhat) ** 2)
            SS_Total = sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1 - (1 - r_squared) * \
                (len(y) - 1) / (len(y) - X.shape[1] - 1)
            
            #BIC
            num_params = len(lin_reg.coef_) + 1
            mse = mean_squared_error(y, yhat)
            bic = calculate_bic(len(y), mse, num_params)

            #results by functional activity across participants 
            results = pd.DataFrame([np.append(coefs, [r_squared, adjusted_r_squared, bic])], columns = results_df.columns)
            results_df = pd.concat([results_df,results], ignore_index=True)

        fname = f'{y_name}_{mask_comb}_regression_results_bysubject_all.csv'
        results_df.to_csv(os.path.join(output_dir, fname), index=False)  


if RUN_DOMINANCE:
    def process_subject(sub, y_name):
        print(f"--- dominance analysis for subject {sub} ----")
        y_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{y_name}_{mask_comb}_effect_size.pickle'), allow_pickle=True).flatten()
        non_nan_indices = ~np.isnan(y_data)
        X = X_data[non_nan_indices, :]
        y = y_data[non_nan_indices]
        m = dominance_stats(X, y)
        with open(os.path.join(output_dir, f'{y_name}_{MASK_NAME}_dominance_sub-{sub:02d}.pickle'), 'wb') as f:
            pickle.dump(m, f)
        total_dominance_array = m["total_dominance"]
        results = pd.DataFrame([total_dominance_array], columns=receptor_names)
        return results

    for y_name in y_names:
        print(f"--- dominance analysis for {y_name} ----")
        results_df = pd.DataFrame(columns=receptor_names)
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_subject, sub, y_name) for sub in subjects]
            for future in futures:
                results = future.result()
                results_df = pd.concat([results_df, results], ignore_index=True)

        # Save data
        results_df.to_pickle(os.path.join(output_dir, f'{y_name}_{mask_comb}_dominance_allsubj.pickle'))
