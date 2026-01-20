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
#specify the number of threads to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import glob
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_1samp
from itertools import chain
import utils.main_funcs as mf
from utils.dominance_funcs import dominance_stats
from config.loader import load_config


# --- Configuration ---
RUN_REGRESSION = True
RUN_DOMINANCE = True


def process_subject(sub, latent_var, task, model_type, output_dir,
                    params, paths, rec):
    print(f"--- Dominance analysis for {task} subject {sub} ----")

    receptor_density = mf.load_receptor_array(paths, rec, on_surface=False)

    y_data = mf.load_effect_map_array(sub, task, latent_var, params, paths)
    non_nan_idx = ~np.isnan(y_data)

    X = receptor_density[non_nan_idx, :]
    y = y_data[non_nan_idx]

    if model_type == 'lin+quad':
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

        # mask out interaction terms with spaces unless "^" present
        mask = [(" " not in n) or ("^" in n) for n in feature_names]
        X = X[:, mask]
        filtered_names = feature_names[mask]

        m = dominance_stats(X, y, feature_names=filtered_names)

    elif model_type == 'linear':
        m = dominance_stats(X, y)

    else:
        raise ValueError(f"Dominance analysis for '{model_type}' not supported!")

    fname = f'{latent_var}_dominance_sub-{sub:02d}_{model_type}.pickle'
    with open(os.path.join(output_dir, fname), 'wb') as f:
        pickle.dump(m, f)

    total_dom = m["total_dominance"]
    return pd.DataFrame([total_dom], columns=rec.receptor_names)

def run_task(task, params, paths, rec, model_type, start_at, num_workers, output_dir):
    if "beta_dir" not in paths:
        beta_dir, add_info = mf.get_beta_dir_and_info(task, params, paths)
    else:
        beta_dir = paths.beta_dir
        add_info = ""

    for latent_var in params.latent_vars:
        files = glob.glob(
            os.path.join(beta_dir, f"sub-*_{latent_var}_effect_size_map{add_info}.nii.gz")
        )
        subjects = sorted({
            Path(f).name.split("_")[0].replace("sub-", "")
            for f in files
        })
        subjects = [int(s) for s in subjects if int(s) not in params.ignore]
        valid_subjects = [s for s in subjects if s > start_at]

        if output_dir == "":
            task_output_dir = os.path.join(beta_dir, 'dominance', rec.source)
            os.makedirs(task_output_dir, exist_ok=True)
        else:
            task_output_dir = output_dir

        print(f"--- Dominance analysis for {task} / {latent_var} ----")

        results_df = pd.DataFrame(columns=rec.receptor_names)

        # SUBJECT-LEVEL PARALLELISM -> workers = 1 if tasks are in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_subject,
                    s, latent_var, task, model_type, task_output_dir,
                    params, paths, rec
                )
                for s in valid_subjects
            ]
            for fut in futures:
                results_df = pd.concat([results_df, fut.result()], ignore_index=True)

        fname = f'{latent_var}_dominance_allsubj_{model_type}.pickle'
        results_df.to_pickle(os.path.join(task_output_dir, fname))


def run_dominance_analysis(params, paths, rec,
                           model_type,
                           start_at,
                           num_workers=4,
                           output_dir=""):
    """
    Runs dominance analysis for all tasks and latent vars in params.
    
    Parameters
    ----------
    params : object
    paths : object
    rec : object
    model_type : str
        'linear' or 'lin+quad'
    start_at : int
        Skip subjects <= this number: in case the analysis was interrupted 
    num_workers : int
        Number of parallel workers
    """

    tasks = params.tasks

    # CASE 1: multiple tasks → parallelize over tasks
    if len(tasks) > 1:
        print("Running tasks in parallel")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_task,
                    task, params, paths, rec,
                    model_type, start_at,
                    1,  # IMPORTANT: avoid nested parallelism
                    output_dir
                )
                for task in tasks
            ]
            for fut in futures:
                fut.result()

    # CASE 2: single task → parallelize over subjects (inside run_task)
    else:
        print("Running subjects in parallel")
        run_task(
            tasks[0], params, paths, rec,
            model_type, start_at,
            num_workers,
            output_dir
        )

def run_regression_analysis(
    params,
    Paths,
    Params,
    rec,
    model_type = 'linear',
):
    """
    Runs the full regression analysis for all tasks and latent variables to inspect or plot their weights

    Parameters
    ----------
    RUN_REGRESSION : bool
        If False, the function exits immediately.
    params : object
        Must contain attributes: tasks, latent_vars, ignore
    Paths : class
        Used to construct path objects: Paths(task=...)
    Params : class
        Used to reinitialize parameter sets per task: Params(task=...)
    rec : object
        Receptor configuration; must contain receptor_names and source.
    model_type : str
        'linear', 'poly2', 'lin+quad', 'lin+interact'
    """

    for task in params.tasks:

        paths_task = Paths(task=task)
        params_task = Params(task=task)

        # === Subject list ===
        fmri_dir = mf.get_fmri_dir(task, paths)
        subjects = [
            s for s in mf.get_subjects(task, fmri_dir)
            if s not in params_task.ignore
        ]

        # Output directory 
        beta_dir, _ = mf.get_beta_dir_and_info(task, params_task, paths_task)
        output_dir = os.path.join(beta_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True)

        # Load receptor density once per task 
        receptor_density = mf.load_receptor_array(
            paths_task, rec, on_surface=False
        )

        # Determine regression feature names
        if model_type == 'linear':
            columns = rec.receptor_names + ["R2", "adjusted_R2"]

        else:
            # Construct polynomial names without transforming large matrices
            poly = PolynomialFeatures(degree=2, include_bias=False)
            dummy = np.zeros((1, receptor_density.shape[1]))
            poly.fit_transform(dummy)
            feature_names = poly.get_feature_names_out(rec.receptor_names)

            if model_type == 'lin+quad':
                # Keep linear + quadratic; drop pure interaction terms
                mask = [(" " not in n) or ("^" in n) for n in feature_names]
            elif model_type == 'lin+interact':
                # Keep linear + interactions; drop squared terms
                mask = ["^" not in n for n in feature_names]
            elif model_type == 'poly2':
                mask = [True] * len(feature_names)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            feature_names = feature_names[mask]
            columns = list(feature_names) + ["R2", "adjusted_R2"]

        # run regression for each latent variable
        for latent_var in params_task.latent_vars:

            results_df = pd.DataFrame(columns=columns)

            for sub in subjects:

                # Load effect map
                y_data = mf.load_effect_map_array(
                    sub, task, latent_var, params_task, paths
                )
                non_nan_idx = ~np.isnan(y_data)

                X = receptor_density[non_nan_idx, :]
                y = y_data[non_nan_idx]

                # Select regression model
                if model_type == 'linear':
                    model = LinearRegression()

                elif model_type == 'poly2':
                    model = make_pipeline(
                        PolynomialFeatures(degree=2, include_bias=False),
                        LinearRegression()
                    )

                else:
                    # Construct polynomial features, then filter them
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    feat_names = poly.get_feature_names_out(
                        input_features=rec.receptor_names
                    )

                    if model_type == 'lin+quad':
                        mask = [(" " not in n) or ("^" in n) for n in feat_names]
                    elif model_type == 'lin+interact':
                        mask = ["^" not in n for n in feat_names]

                    X = X_poly[:, mask]
                    model = LinearRegression()

                # ----------------------------------------------------------
                # Fit model
                # ----------------------------------------------------------
                model.fit(X, y)

                if model_type == 'poly2':
                    coefs = model.named_steps['linearregression'].coef_
                else:
                    coefs = model.coef_

                # ----------------------------------------------------------
                # Compute R² and adjusted R²
                # ----------------------------------------------------------
                yhat = model.predict(X)

                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)

                r2 = 1 - ss_res / ss_tot
                adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

                row = np.append(coefs, [r2, adj_r2])
                results_df.loc[len(results_df)] = row

            # ==================================================================
            # Save results
            # ==================================================================
            fname = f"{latent_var}_regression_weights_bysubject_all_{model_type}.csv"
            results_df.to_csv(os.path.join(output_dir, fname), index=False)


def plot_regression_coefficients(tasks, model_type='linear'):
    """Plot regression coefficients across subjects and save mean coefficients table."""
    
    # ----container for mean coefficients ----
    mean_coeffs_all_studies = []

    for task in tasks:
        params, paths, rec = load_config(task, return_what='all')

        if 'beta_dir' in paths:
            beta_dir = paths.beta_dir
        else: 
            beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)

        data_dir = os.path.join(beta_dir, 'regressions', rec.source)

        output_dir = os.path.join(paths.out_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True)

        base_colors = sns.color_palette('husl', len(rec.receptor_groups))
        plt.rcParams.update({'font.size': 18})

        for latent_var in params.latent_vars:
            fname = f'{latent_var}_regression_results_bysubject_all_{model_type}.csv'
            file_path = os.path.join(data_dir, fname)
            if not os.path.exists(file_path):
                print(f"Skipping {latent_var} — no file found.")
                continue

            results_df = pd.read_csv(file_path)
            if 'a2' in results_df.columns:
                results_df.rename(columns={'a2': 'A2'}, inplace=True)

            # ---- compute mean coefficients for this study ----
            mean_coeffs = results_df[rec.receptor_names].mean()
            mean_coeffs.name = task  # row label = study
            mean_coeffs_all_studies.append(mean_coeffs)

            # --- T-tests per receptor ---
            t_values, p_values = [], []
            for receptor in rec.receptor_names:
                t, p = ttest_1samp(results_df[receptor], 0)
                t_values.append(t)
                p_values.append(p)

            _, p_corr = fdrcorrection(p_values, alpha=0.05)
            sig_receptors = [r for r, pc in zip(rec.receptor_names, p_corr) if pc < 0.05]
            sig_signs = [np.sign(t) for t, pc in zip(t_values, p_corr) if pc < 0.05]

            # --- Group color logic ---
            receptor_to_group = {r: i for i, grp in enumerate(rec.receptor_groups) for r in grp}
            receptor_to_class = {r: i for i, grp in enumerate(rec.receptor_class) for r in grp}
            ordered = [r for grp in rec.receptor_groups for r in grp]

            colors = []
            for receptor in ordered:
                g = receptor_to_group.get(receptor, -1)
                c = receptor_to_class.get(receptor, -1)
                if c == 0:
                    col = sns.dark_palette(base_colors[g], n_colors=3)[1]
                elif c == 1:
                    col = sns.light_palette(base_colors[g], n_colors=3)[1]
                else:
                    col = sns.light_palette(base_colors[g], n_colors=3)[0]
                colors.append(col)

            # --- Plot ---
            fig, ax = plt.subplots() 
            sns.boxplot(data=results_df[ordered], ax=ax, palette=colors)
            for i, receptor in enumerate(ordered):
                if receptor in sig_receptors:
                    idx = sig_receptors.index(receptor)
                    color = '#C96868' if sig_signs[idx] > 0 else '#7EACB5'
                    ax.scatter(i, results_df[receptor].median(), color=color, zorder=5)

            ax.set_xticklabels(rec.receptor_label_formatted, rotation=90)
            ax.set_xlabel('Receptor')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'{latent_var} regression coefficients ({rec.source})')

            plt.tight_layout()
            fig_dir = os.path.join(output_dir, 'plots')
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f'{latent_var}_coefficients_{model_type}.png'), dpi=300)
            plt.close()

    # ---- save combined table ----
    mean_coeffs_df = pd.DataFrame(mean_coeffs_all_studies)
    mean_coeffs_df.index.name = 'study'

    csv_path = os.path.join(output_dir, f'mean_regression_coefficients_{model_type}.csv')
    mean_coeffs_df.to_csv(csv_path)


# ---------- Data loading ----------
def load_dominance_data(params, paths, latent_var, model_type="linear"):
    """
    Loads and standardizes dominance results for given experiments.
    
    Returns
    -------
    results : dict if more than one task, df otherwise
        {task_name: standardized_df}
    """
    results = {}
    model_suffix = "" if model_type == "linear" else "_lin+quad"

    for task in params.tasks:

        # Determine input directory
        if 'beta_dir' not in paths:
            beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
            input_dir = os.path.join(beta_dir, "dominance", "PET2")
        else:
            input_dir = os.path.join(paths.results_dir, "dominance")

        fname = f"{latent_var}_dominance_allsubj{model_suffix}.pickle"
        df = pd.read_pickle(os.path.join(input_dir, fname))

        # Column consistency
        if "a2" in df.columns:
            df = df.rename(columns={"a2": "A2"})

        # Row-wise standardization
        standardized_df = df.div(df.sum(axis=1), axis=0)

        results[task] = standardized_df

    return results


# ---------- Aggregation ----------
def aggregate_dominance(results_dict, exclude_explore=False):
    """
    Aggregates dominance data across experiments.
    Returns a combined DataFrame (concatenated subjects across studies)
    and a mean per-study DataFrame for heatmaps.
    """
    if exclude_explore:
        results_dict = {k: v for k, v in results_dict.items() if k != "Explore"}

    combined = pd.concat(results_dict.values(), ignore_index=True)
    per_study_means = {k: v.mean() for k, v in results_dict.items()}
    return combined, pd.DataFrame(per_study_means).T


# ---------- Plotting ----------
def plot_dominance_bars(
    df, receptor_groups, receptor_class, receptor_label_formatted,
    title=None, show_errorbars=True, ylim=None
):
    """
    Generic barplot function for dominance data.
    Works for both individual studies and group-level averages.
    """
    # Order and maps
    ordered_receptors = [
    r for group in receptor_groups for r in group
    if isinstance(r, str)]
    receptor_to_group = {r: i for i, g in enumerate(receptor_groups) for r in g}
    receptor_to_class = {r: i for i, g in enumerate(receptor_class) for r in g}

    # Base color palette for groups
    base_colors = sns.color_palette("husl", len(receptor_groups))

    colors = []
    for receptor in ordered_receptors:
        group_idx = receptor_to_group.get(receptor, -1)
        class_type = receptor_to_class.get(receptor, -1)
        if class_type == 0:  # Excitatory
            color = sns.dark_palette(base_colors[group_idx], n_colors=3)[1]
            colors.append({"face": color, "edge": color})
        elif class_type == 1:  # Inhibitory
            color = sns.light_palette(base_colors[group_idx], n_colors=3)[1]
            colors.append({"face": color, "edge": color})
        else:  # “other” receptors
            face_color = sns.light_palette(base_colors[group_idx], n_colors=3)[0]
            edge_color = sns.dark_palette(base_colors[group_idx], n_colors=3)[2]
            colors.append({"face": face_color, "edge": edge_color})

    # Compute mean & SEM 
    mean_vals = df[ordered_receptors].mean()
    sem_vals = df[ordered_receptors].sem()


    # --- PLOT ---
    fig, ax = plt.subplots() 

    bars = ax.bar(
        ordered_receptors,
        mean_vals,
        yerr=sem_vals if show_errorbars else None,
        color=[c["face"] for c in colors],
        edgecolor=[c["edge"] for c in colors],
    )

    # Add hatching for receptors not in excitatory/inhibitory classes
    for i, receptor in enumerate(ordered_receptors):
        if receptor not in receptor_class[0] and receptor not in receptor_class[1]:
            bars[i].set_hatch("//")

    # --- Formatting ---
    ax.set_xticks(np.arange(len(ordered_receptors)))
    ax.set_xticklabels(receptor_label_formatted, rotation=90)
    for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
        group_idx = receptor_to_group.get(receptor, -1)
        label.set_color(base_colors[group_idx])

    ax.set_ylabel("% Contribution")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    return fig, ax

def plot_dominance_heatmap(
    all_means,
    receptor_groups,
    cmap,
    receptor_label_formatted,
    title=None,
    rename_tasks=False,
    params = None,
):
    """
    Generic heatmap plotting for dominance means across studies.
    
    Parameters
    ----------
    all_means : pd.DataFrame
        DataFrame of dominance means (rows=studies/tasks, columns=receptors)
    receptor_groups : list of lists
        Groups of receptors to determine order
    cmap : matplotlib colormap
        Colormap for heatmap
    receptor_label_formatted : list
        Labels for x-axis receptors
    title : str, optional
        Figure title
    rename_tasks : bool, default False
        If True, remap row/column labels containing task names to study names
        according to `params.study_mapping`.
    """
    ordered_receptors = [r for group in receptor_groups for r in group]
    
    all_means_to_plot = all_means.copy()

    # Apply task-to-study remapping if requested
    if rename_tasks:
        new_index = [
            next((params.study_mapping[t] for t in params.study_mapping if t in str(lbl)), lbl)
            for lbl in all_means_to_plot.index
        ]
        all_means_to_plot.index = new_index

    fig, ax = plt.subplots(figsize=(3.38, 1.5))
    vmin=0
    vmax=0.18
    ax = sns.heatmap(
        all_means_to_plot[ordered_receptors],
        xticklabels=receptor_label_formatted,
        cmap=cmap,
        linewidths=1,
        vmin=vmin,
        vmax=vmax,
        cbar = False
    )

    plt.yticks(rotation=45)

    if title:
        plt.title(title)
    plt.tight_layout()

    return fig, ax



def plot_separate_colorbar(cmap, vmin=0, vmax=0.18, label="Contribution (%)", orientation="vertical"):
    # Create the figure and axis for the colorbar
    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(0.15, 1.3))
    else:
        fig, ax = plt.subplots(figsize=(1.5, 0.25))
    
    # Create a ScalarMappable to serve as colorbar data source
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Draw the colorbar on the axis
    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
    cbar.set_label(label, labelpad=-10)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    
    if orientation == "vertical":
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.label.set_verticalalignment('center')
    else:
        cbar.ax.xaxis.set_label_position('bottom')

    return fig, ax

def plot_legend_dominance_bars(rec, ncol=None, fig_width=8, fig_height=1.2):
    """
    Create a standalone legend spanning multiple columns, suitable for placing
    above combined plots (e.g., in Inkscape).
    """
 # --- Define colors and elements ---
    base_colors = sns.color_palette('husl', len(rec.group_names))
    legend_elements = []
    legend_labels = []

    for color, name in zip(base_colors, rec.group_names):
        dark_color = sns.dark_palette(color, n_colors=3)[1]
        light_color = sns.light_palette(color, n_colors=3)[1]

        dark_square = mlines.Line2D([], [], color=dark_color, marker='s', markersize=7, linestyle='None')
        light_square = mlines.Line2D([], [], color=light_color, marker='s', markersize=7, linestyle='None')

        legend_elements.append((dark_square, light_square))
        legend_labels.append(name)

    # Excitatory/inhibitory
    dark_grey_patch = mpatches.Patch(color="dimgray", label="excitatory")
    light_grey_patch = mpatches.Patch(color="lightgrey", label="inhibitory")
    legend_elements.extend([dark_grey_patch, light_grey_patch])
    legend_labels.extend(["excitatory", "inhibitory"])

    # Transporter hatch
    hatch_example = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='transporter')
    legend_elements.append(hatch_example)
    legend_labels.append("transporter")
    
    # --- Plot legend only ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.legend(
        legend_elements,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},
        loc="center",
        ncol=ncol,             # number of columns across
        frameon=False,
        columnspacing=1.0,     # 🔹 space between columns
        handlelength=2.5,      # 🔹 keeps paired squares apart from text
        handletextpad=0.5,     # 🔹 space between handles and labels
        labelspacing=1.0,      # vertical spacing between rows (if wrapping)
    )

    return fig


if __name__ == "__main__":
    params, paths, rec = load_config('all', return_what='all')
    mf.set_publication_style(font_size=8)

    if RUN_REGRESSION:
        # Regression Analysis
        run_regression_analysis(
            RUN_REGRESSION=True,
            params=params,
            Paths=paths,
            Params=params,
            rec=rec,
            MODEL_TYPE="linear"
        )

    #individual regression plots
    plot_regression_coefficients(params.tasks, model_type='linear')
          

    # Dominance Analysis
    if RUN_DOMINANCE:

        run_dominance_analysis(
            params=params,
            paths=paths,
            rec=rec,
            model_type="linear",
            start_at=0,
            num_workers=4
        )

    for latent_var in params.latent_vars:

        # Load all dominance results
        results = load_dominance_data(params, paths, latent_var, model_type='linear')

        plot_dir = os.path.join(paths.home_dir, "figures")
        # ---- Individual plots ----
        for exp, df in results.items():
            title = f"{exp} – {latent_var}"
            fig, ax = plot_dominance_bars(df, params.receptor_groups, params.receptor_class,
                                title=title)
            mf.save_figure(fig, plot_dir, f"{exp}_{latent_var}_dominance")

        # ---- Group-level mean  ----
        combined, per_study_means = aggregate_dominance(results)

        fig, ax = plot_dominance_bars(combined, rec.receptor_groups, rec.receptor_class, rec.receptor_label_formatted, ylim=(0, 0.16))
        mf.save_figure(fig, plot_dir, f"group_{latent_var}_dominance")

        # ---- Heatmaps ----
        cmap_pos = mf.get_custom_colormap('pos')
        fig, ax = plot_dominance_heatmap(per_study_means, rec.receptor_groups, cmap_pos, rec.receptor_label_formatted, rename_tasks=True, params=params, paths=paths)
        mf.save_figure(fig, plot_dir, f"heatmap_{latent_var}")


    # ---- Legend for dominance barplot ----
    fig = plot_legend_dominance_bars(rec)
    mf.save_figure(fig, plot_dir, f"legend_{latent_var}_dominance_bar")
