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
import sys
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
from params_and_paths import Paths, Params, Receptors
from dominance_funcs import dominance_stats
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import main_funcs as mf

# --- Configuration ---
MODEL_TYPE = 'linear'      # Options: 'linear', 'lin+quad', 'lin+interact', 'poly2'
RUN_REGRESSION = True
RUN_DOMINANCE = True
NUM_WORKERS = 30           # Parallel dominance analysis workers
START_AT = 0               # Resume point for dominance analysis

# --- Initialize paths and parameters ---
rec =Receptors(source = 'PET2')


def plot_regression_coefficients(tasks, model_type='linear'):
    """Plot regression coefficients across subjects with FDR-corrected significance."""
    
    for task in tasks:
        paths = Paths(task=task)
        params = Params(task=task)

        beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
        output_dir = os.path.join(beta_dir, 'regressions', rec.source)
        os.makedirs(output_dir, exist_ok=True)


        base_colors = sns.color_palette('husl', len(params.receptor_groups))
        plt.rcParams.update({'font.size': 18})

        for latent_var in params.latent_vars:
            fname = f'{latent_var}_{params.mask}_regression_results_bysubject_all_{model_type}.csv'
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

            # --- Group color logic ---
            receptor_to_group = {r: i for i, grp in enumerate(params.receptor_groups) for r in grp}
            receptor_to_class = {r: i for i, grp in enumerate(params.receptor_class) for r in grp}
            ordered = [r for grp in params.receptor_groups for r in grp]

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
            fig, ax = plt.subplots() 
            sns.boxplot(data=results_df[ordered], ax=ax, palette=colors)
            for i, receptor in enumerate(ordered):
                if receptor in sig_receptors:
                    idx = sig_receptors.index(receptor)
                    color = '#C96868' if sig_signs[idx] > 0 else '#7EACB5'
                    ax.scatter(i, results_df[receptor].median(), color=color, zorder=5)

            ax.set_xticklabels(params.receptor_label_formatted, rotation=90)
            ax.set_xlabel('Receptor')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'{latent_var} regression coefficients ({rec.source})')

            plt.tight_layout()
            fig_dir = os.path.join(output_dir, 'plots')
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f'{latent_var}_coefficients_{model_type}.png'), dpi=300)
            plt.close()

# ---------- Data loading ----------
def load_dominance_data(tasks, latent_var, model_type="linear"):
    """
    Loads and standardizes dominance results for given experiments.
    Returns a dictionary {exp_name: standardized_df}
    """
    results = {}
    model_suffix = "" if model_type == "linear" else "_lin+quad"

    for task in tasks:
        paths = Paths(task=tasks)
        params = Params(task=task)
        beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
        input_dir = os.path.join(beta_dir, "regressions", "PET2")
        fname = f"{latent_var}_{params.mask}_dominance_allsubj{model_suffix}.pickle"
        df = pd.read_pickle(os.path.join(input_dir, fname))
        if "a2" in df.columns:
            df.rename(columns={"a2": "A2"}, inplace=True)
        standardized_df = df.div(df.sum(axis=1), axis=0)
        results[task] = standardized_df
    return results


# ---------- Aggregation ----------
def aggregate_dominance(results_dict, exclude_explore=True):
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
    title=None, show_errorbars=True, ylim=(0, 0.16)
):
    """
    Generic barplot function for dominance data.
    Works for both individual studies and group-level averages.
    """
    # Order and maps
    ordered_receptors = [r for group in receptor_groups for r in group]
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

    ax.set_xlabel("Receptor/Transporter")
    ax.set_ylabel("% Contribution")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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

    fig, ax = plt.subplots(figsize=(3.3, 1.5))
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
        # cbar_kws=dict(location="right", label="% Contribution")
    )
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([vmin, vmax])
    # cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    # cbar.ax.tick_params(pad=1)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # ha='right' aligns them nicely
    plt.yticks(rotation=45)

    # cbar.set_label("Contribution (%)", labelpad=-10) 
    # cbar.ax.yaxis.set_label_position('left')          
    # cbar.ax.yaxis.label.set_verticalalignment('center')

    if title:
        plt.title(title)
    plt.tight_layout()

    return fig, ax

def plot_explore_dominance_heatmap(
    latent_vars,
    receptor_groups,
    receptor_label_formatted,
    cmap,
    model_type="linear",
    title=None,
    params = None,
    paths = None,
):
    """
    Special-case function to plot Explore dominance heatmap
    including both latent variables in a single plot.
    """

    # Load dominance data only for Explore
    all_explore_means = []
    model_suffix = "" if model_type == "linear" else "_lin+quad"
    task = "Explore"

    ordered_receptors = [r for group in receptor_groups for r in group]

    for lv in latent_vars:
        beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
        input_dir = os.path.join(beta_dir, "regressions", "PET2")

        fname = f"{lv}_{params.mask}_dominance_allsubj{model_suffix}.pickle"
        df = pd.read_pickle(os.path.join(input_dir, fname))

        if "a2" in df.columns:
            df.rename(columns={"a2": "A2"}, inplace=True)

        # Standardize
        df = df.div(df.sum(axis=1), axis=0)

        mean_vals = df[ordered_receptors].mean()
        mean_vals.name = lv
        all_explore_means.append(mean_vals)

    # Create row × receptor matrix (rows = latent vars)
    explore_means = pd.concat(all_explore_means, axis=1).T

    vmin=0
    vmax=0.18

    fig, ax = plt.subplots(figsize=(3.3, 1.5))

    # Plot using existing helper heatmap function
    ax = sns.heatmap(
        explore_means[ordered_receptors],
        xticklabels=receptor_label_formatted,
        cmap=cmap,
        linewidths=1,
        vmin=vmin,
        vmax=vmax,
        cbar = False
    )
    # cbar = ax.collections[0].colorbar
    # cbar.set_ticks([vmin, vmax])
    # cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    # cbar.ax.tick_params(pad=1)

    # cbar.set_label("Contribution (%)", labelpad=-10)  
    # cbar.ax.yaxis.set_label_position('left')        
    # cbar.ax.yaxis.label.set_verticalalignment('center')

    plt.yticks(rotation=45)

    return fig, ax

def plot_separate_colorbar(cmap, vmin=0, vmax=0.18, label="Contribution (%)", orientation="vertical"):
    # Create the figure and axis for the colorbar
    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(0.25, 1.3))
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

    plt.tight_layout()
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

        dark_square = mlines.Line2D([], [], color=dark_color, marker='s', markersize=10, linestyle='None')
        light_square = mlines.Line2D([], [], color=light_color, marker='s', markersize=10, linestyle='None')

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

    # Significance markers
    red_dot = mlines.Line2D(
        [], [], 
        color='black', 
        marker='+', 
        markersize=8,          # make it clearly visible
        markeredgewidth=1.5,    # thicker lines for visibility
        linestyle='None',
        label="sig. estimate in full model: positive"
    )
    blue_dot = mlines.Line2D(
        [], [], 
        color='black', 
        marker='_', 
        markersize=10,          # slightly larger for underscore visibility
        markeredgewidth=2, 
        linestyle='None',
        label="sig. estimate in full model: negative"
    )
    legend_elements.extend([red_dot, blue_dot])
    legend_labels.extend([
        "sig. estimate in full model: positive",
        "sig. estimate in full model: negative"
    ])

    # --- Plot legend only ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    legend = ax.legend(
        legend_elements,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},
        loc="center",
        ncol=ncol,             # number of columns across
        frameon=False,
        columnspacing=3.0,     # 🔹 space between columns
        handlelength=4.0,      # 🔹 keeps paired squares apart from text
        handletextpad=1.0,     # 🔹 space between handles and labels
        labelspacing=1.5,      # vertical spacing between rows (if wrapping)
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    params = Params(task='all')
    # Regression Analysis
    if RUN_REGRESSION:
        for task in params.tasks:
            paths = Paths(task=task)
            params = Params(task=task)
            fmri_dir = mf.get_fmri_dir(task, paths)
            subjects = [s for s in mf.get_subjects(task, fmri_dir) if s not in params.ignore]      
            beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
            output_dir = os.path.join(beta_dir, 'regressions', rec.source)
            os.makedirs(output_dir, exist_ok=True)

            receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)
            receptor_density = mf.load_receptor_array(on_surface=False)

            # Determine regression columns
            if MODEL_TYPE == 'linear':
                columns = rec.receptor_names + ["R2", "adjusted_R2"]
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

                columns = list(feature_names) + ["R2", "adjusted_R2",]

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

            beta_dir, _ = mf.get_beta_dir_and_info(task, params, paths)
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
            fmri_dir = mf.get_fmri_dir(task, paths)
            subjects = [s for s in mf.get_subjects(task, fmri_dir) if s not in params.ignore]
            for latent_var in params.latent_vars:
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

    mf.set_publication_style(font_size=8)

    #individual regression plots
    plot_regression_coefficients(params.tasks, model_type=MODEL_TYPE)

    for latent_var in params.latent_vars:

        # Load all dominance results
        results = load_dominance_data(params.tasks, latent_var, model_type=MODEL_TYPE)

        plot_dir = os.path.join(paths.home_dir, "figures")
        # ---- Individual plots ----
        for exp, df in results.items():
            title = f"{exp} – {latent_var}"
            fig, ax = plot_dominance_bars(df, params.receptor_groups, params.receptor_class,
                                title=title)
            mf.save_figure(fig, plot_dir, f"{exp}_{latent_var}_dominance")

        # ---- Group-level mean (exclude Explore) ----
        combined, per_study_means = aggregate_dominance(results, exclude_explore=True)
        fig, ax = plot_dominance_bars(combined, rec.receptor_groups, rec.receptor_class, rec.receptor_label_formatted)
        mf.save_figure(fig, plot_dir, f"group_{latent_var}_dominance")

        # ---- Heatmaps ----
        cmap_pos = mf.get_custom_colormap('pos')
        fig, ax = plot_dominance_heatmap(per_study_means, rec.receptor_groups, cmap_pos, rec.receptor_label_formatted, rename_tasks=True, params=params, paths=paths)
        mf.save_figure(fig, plot_dir, f"heatmap_{latent_var}")

    # ---- Explore-only heatmap ----
    paths = Paths(task='Explore')
    params = Params(task='Explore')
    fig, ax = plot_explore_dominance_heatmap(
        params.latent_vars,
        rec.receptor_groups,
        rec.receptor_label_formatted,
        cmap_pos,
        model_type=MODEL_TYPE,
        params=params,
        paths=paths
    )
    mf.save_figure(fig, plot_dir, "heatmap_explore")

    # ---- Legend for dominance barplot ----
    fig = plot_legend_dominance_bars(rec)
    mf.save_figure(fig, plot_dir, f"legend_{latent_var}_dominance_bar")
