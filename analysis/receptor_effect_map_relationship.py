#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:25:09 2024
@author: Alice Hodapp

This script retrieves receptor density data and GLM output to run multiple regression
and/or dominance analysis separately for each subject and latent variable
(surprise, confidence, predictability). Outputs include regression coefficients
and total dominance values.
"""

import os
# Limit resource usage before importing NumPy
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

# --- Configuration ---
MODEL_TYPE = 'linear'      # Options: 'linear', 'lin+quad', 'lin+interact', 'poly2'
RUN_REGRESSION = True
RUN_DOMINANCE = True
NUM_WORKERS = 30           # Parallel dominance analysis workers
START_AT = 0               # Resume point for dominance analysis

# --- Initialize paths and parameters ---
paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = [s for s in mf.get_subjects(params.db, fmri_dir) if s not in params.ignore]

def plot_regression_coefficients(rec, params, paths, mask_comb, proj='', model_type='linear'):
    """Plot regression coefficients across subjects with FDR-corrected significance."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection

    beta_dir = os.path.join(paths.home_dir, params.db, params.mask, 'first_level')
    output_dir = os.path.join(beta_dir, 'regressions', rec.source)
    os.makedirs(output_dir, exist_ok=True)

    if params.db == 'Explore':
        variables = ['surprise', 'confidence']
    else:
        variables = params.latent_vars

    # Grouping and color setup
    if rec.source in ['PET', 'PET2']:
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline,
                           rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.cannabinnoid]
    elif rec.source == 'autorad_zilles44':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline,
                           rec.glutamate, rec.gaba, rec.dopamine]
    elif rec.source == 'AHBA':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.dopamine]
    receptor_class = [rec.exc, rec.inh]

    base_colors = sns.color_palette('husl', len(receptor_groups))
    plt.rcParams.update({'font.size': 18})

    receptor_label_formatted = [
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$', '$5\\text{-}\\mathrm{HTT}$',
        '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$', '$\\mathrm{M}_{\\mathrm{1}}$',
        '$\\mathrm{VAChT}$', '$\\mathrm{NET}$', '$\\mathrm{A}_{\\mathrm{2}}$',
        '$\\mathrm{MOR}$', '$\\mathrm{mGluR}_{\\mathrm{5}}$', '$\\mathrm{NMDA}$',
        '$\\mathrm{H}_{\\mathrm{3}}$', '$\\mathrm{GABA}_{\\mathrm{a}}$', '$\\mathrm{D}_{\\mathrm{1}}$',
        '$\\mathrm{D}_{\\mathrm{2}}$', '$\\mathrm{DAT}$', '$\\mathrm{CB}_{\\mathrm{1}}$'
    ]

    for latent_var in variables:
        fname = f'{latent_var}_{mask_comb}_regression_results_bysubject_all{proj}_{model_type}.csv'
        file_path = os.path.join(output_dir, fname)
        if not os.path.exists(file_path):
            print(f"Skipping {latent_var} — no file found.")
            continue

        results_df = pd.read_csv(file_path)
        if 'a2' in results_df.columns:
            results_df.rename(columns={'a2': 'A2'}, inplace=True)

        # --- T-tests per receptor ---
        t_values, p_values = [], []
        for receptor in rec.receptor_names:
            t, p = ttest_1samp(results_df[receptor], 0)
            t_values.append(t)
            p_values.append(p)

        _, p_corr = fdrcorrection(p_values, alpha=0.05)
        sig_receptors = [r for r, pc in zip(rec.receptor_names, p_corr) if pc < 0.05]
        sig_signs = [np.sign(t) for t, pc in zip(t_values, p_corr) if pc < 0.05]

        mean_R2 = results_df['R2'].mean()
        mean_BIC = results_df['BIC'].mean()

        # --- Group color logic ---
        receptor_to_group = {r: i for i, grp in enumerate(receptor_groups) for r in grp}
        receptor_to_class = {r: i for i, grp in enumerate(receptor_class) for r in grp}
        ordered = [r for grp in receptor_groups for r in grp]

        colors = []
        for receptor in ordered:
            g = receptor_to_group.get(receptor, -1)
            c = receptor_to_class.get(receptor, -1)
            if c == 0:  # Excitatory
                col = sns.dark_palette(base_colors[g], n_colors=3)[1]
            elif c == 1:  # Inhibitory
                col = sns.light_palette(base_colors[g], n_colors=3)[1]
            else:
                col = sns.light_palette(base_colors[g], n_colors=3)[0]
            colors.append(col)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=results_df[ordered], ax=ax, palette=colors)
        for i, receptor in enumerate(ordered):
            if receptor in sig_receptors:
                idx = sig_receptors.index(receptor)
                color = '#C96868' if sig_signs[idx] > 0 else '#7EACB5'
                ax.scatter(i, results_df[receptor].median(), color=color, zorder=5)

        ax.set_xticklabels(receptor_label_formatted, rotation=90)
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{latent_var} regression coefficients ({rec.source})')

        textstr = f'Mean R²: {mean_R2:.2f}\nMean BIC: {mean_BIC:.2f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fig_dir = os.path.join(output_dir, 'plots')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f'{latent_var}_coefficients_{model_type}.png'), dpi=300)
        plt.close()

def plot_dominance_results(rec, params, paths, mask_comb, model_type='lin+quad'):
    """Plot mean dominance contributions across receptors with group colors."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    beta_dir = os.path.join(paths.home_dir, params.db, params.mask, 'first_level')
    output_dir = os.path.join(beta_dir, 'regressions', rec.source)
    os.makedirs(output_dir, exist_ok=True)

    variables = ['surprise', 'confidence'] if params.db == 'Explore' else params.latent_vars

    if rec.source in ['PET', 'PET2']:
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline,
                           rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.cannabinnoid]
    elif rec.source == 'autorad_zilles44':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline,
                           rec.glutamate, rec.gaba, rec.dopamine]
    elif rec.source == 'AHBA':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.dopamine]
    receptor_class = [rec.exc, rec.inh]

    base_colors = sns.color_palette('husl', len(receptor_groups))
    receptor_to_group = {r: i for i, grp in enumerate(receptor_groups) for r in grp}
    receptor_to_class = {r: i for i, grp in enumerate(receptor_class) for r in grp}
    ordered = [r for grp in receptor_groups for r in grp]

    for latent_var in variables:
        file_path = os.path.join(output_dir, f'{latent_var}_{mask_comb}_dominance_allsubj_{model_type}.pickle')
        if not os.path.exists(file_path):
            print(f"Skipping {latent_var} — no dominance file found.")
            continue

        results_df = pd.read_pickle(file_path)
        if 'a2' in results_df.columns:
            results_df.rename(columns={'a2': 'A2'}, inplace=True)

        # Normalize by total contribution
        standardized = results_df[ordered].div(results_df[ordered].sum(axis=1), axis=0)
        bar_means = standardized.mean()
        bar_sem = standardized.sem()

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(ordered, bar_means, yerr=bar_sem,
                      color=[sns.dark_palette(base_colors[receptor_to_group[r]], n_colors=3)[1]
                             if receptor_to_class.get(r, 0) == 0
                             else sns.light_palette(base_colors[receptor_to_group[r]], n_colors=3)[1]
                             for r in ordered],
                      capsize=5)
        ax.set_xticklabels(ordered, rotation=90)
        ax.set_xlabel('Receptor/Transporter')
        ax.set_ylabel('Contribution (%)')
        plt.tight_layout()

        fig_dir = os.path.join(output_dir, 'plots')
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f'{latent_var}_dominance_{model_type}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Regression Analysis
if RUN_REGRESSION:
    for task in params.tasks:
        beta_dir, _ = mf.get_beta_dir_and_info()
        output_dir = os.path.join(beta_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True)

        receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)
        receptor_density = mf.load_recptor_array(on_surface=False)

        # Determine regression columns
        if MODEL_TYPE == 'linear':
            columns = rec.receptor_names + ["R2", "adjusted_R2", "BIC"]
        else:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            dummy = np.zeros((1, receptor_density.shape[1]))
            poly.fit_transform(dummy)
            feature_names = poly.get_feature_names_out(rec.receptor_names)

            if MODEL_TYPE == 'lin+quad':
                # Linear + quadratic only (exclude interaction terms)
                mask = [(" " not in n) or ("^" in n) for n in feature_names]
                feature_names = feature_names[mask]
            elif MODEL_TYPE == 'lin+interact':
                # Linear + interactions only (exclude squared terms)
                mask = ["^" not in n for n in feature_names]
                feature_names = feature_names[mask]

            columns = list(feature_names) + ["R2", "adjusted_R2", "BIC"]

        for latent_var in params.latent_vars:
            results_df = pd.DataFrame(columns=columns)

            for sub in subjects:
                y_data = mf.load_effect_map_array(sub, task, latent_var)
                receptor_density_zm = receptor_density

                non_nan_idx = ~np.isnan(y_data)
                X = receptor_density_zm[non_nan_idx, :]
                y = y_data[non_nan_idx]

                # Model selection
                if MODEL_TYPE == 'linear':
                    model = LinearRegression()
                elif MODEL_TYPE == 'poly2':
                    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                                          LinearRegression())
                else:
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X = poly.fit_transform(X)
                    feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

                    if MODEL_TYPE == 'lin+quad':
                        mask = [(" " not in n) or ("^" in n) for n in feature_names]
                    elif MODEL_TYPE == 'lin+interact':
                        mask = ["^" not in n for n in feature_names]

                    X = X[:, mask]
                    filtered_feature_names = feature_names[mask]
                    model = LinearRegression()

                # Fit model and compute metrics
                model.fit(X, y)
                coefs = (model.named_steps['linearregression'].coef_
                         if MODEL_TYPE == 'poly2' else model.coef_)

                yhat = model.predict(X)
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot
                adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

                results = pd.DataFrame([np.append(coefs, [r2, adj_r2])],
                                       columns=columns)
                results_df = pd.concat([results_df, results], ignore_index=True)

            fname = f'{latent_var}_{params.mask}_regression_results_bysubject_all_{MODEL_TYPE}.csv'
            results_df.to_csv(os.path.join(output_dir, fname), index=False)

# Dominance Analysis
if RUN_DOMINANCE:

    def process_subject(sub, latent_var, task):
        """Run dominance analysis for a single subject."""
        print(f"--- Dominance analysis for {task} subject {sub} ----")

        beta_dir, _ = mf.get_beta_dir_and_info()
        output_dir = os.path.join(beta_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True)

        y_data = mf.load_effect_map_array(sub, task, latent_var)
        non_nan_idx = ~np.isnan(y_data)
        X = receptor_density[non_nan_idx, :]
        y = y_data[non_nan_idx]

        if MODEL_TYPE == 'lin+quad':
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)
            mask = [(" " not in n) or ("^" in n) for n in feature_names]
            X = X[:, mask]
            filtered_feature_names = feature_names[mask]
            m = dominance_stats(X, y, feature_names=filtered_feature_names)
        elif MODEL_TYPE == 'linear':
            m = dominance_stats(X, y)
        else:
            raise ValueError(f"Dominance analysis for '{MODEL_TYPE}' not supported!")

        fname = f'{latent_var}_{params.mask}_dominance_sub-{sub:02d}_{MODEL_TYPE}.pickle'
        with open(os.path.join(output_dir, fname), 'wb') as f:
            pickle.dump(m, f)

        total_dominance = m["total_dominance"]
        return pd.DataFrame([total_dominance], columns=rec.receptor_names)

    for task in params.tasks:
        for latent_var in params.variables:
            print(f"--- Dominance analysis for {latent_var} ----")
            results_df = pd.DataFrame(columns=rec.receptor_names)
            valid_subjects = [s for s in subjects if s > START_AT]

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(process_subject, s, latent_var, task)
                           for s in valid_subjects]
                for future in futures:
                    res = future.result()
                    results_df = pd.concat([results_df, res], ignore_index=True)

            # Save combined results
            fname = f'{latent_var}_{params.mask}_dominance_allsubj_{MODEL_TYPE}.pickle'
            results_df.to_pickle(os.path.join(output_dir, fname))



if __name__ == "__main__":
