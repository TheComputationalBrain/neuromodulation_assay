import os
import glob
import sys
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
from pathlib import Path
from neuromaps import transforms
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore, sem, ttest_rel
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nilearn import datasets
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors

#analysis
FROM_BETA = True
COMP_NULL = False
COMPARE_LANG_LEARN = False
COMPARE_EXPL_VAR = False

#plots
PLOT_VAR_EXPLAINED = False
PLOT_VAR_EXPLAINED_RATIO = False

MODEL_TYPE = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'

SCORE = 'determination'

suffix = ''

paths = Paths(task='all')
params = Params(task='all')
rec = Receptors()

output_dir = os.path.join(paths.home_dir, 'variance_explained')
os.makedirs(output_dir, exist_ok=True) 

comparison_latent = 'language'
comparison_task = 'lanA'

fsavg = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

if FROM_BETA:
    for task in params.tasks: 

        beta_dir, add_info = mf.get_beta_dir_and_info(task, params, paths)
        fmri_dir = mf.get_fmri_dir(task)

        subject_paths = paths.home_dir if task == "lanA" else paths.root_dir
        subjects = mf.get_subjects(task, os.path.join(subject_paths, fmri_dir))
        subjects = [subj for subj in subjects if subj not in params.ignore[task]] 

        with open(os.path.join(output_dir,f'predict_from_beta.txt'), "a") as outfile:
            outfile.write(f'{task}: variance explained in analysis:\n\n')
            
            for latent_var in params.latent_vars:
                all_data = []

                #load all data
                for sub in subjects:
                    if task == 'lanA':
                        file = nib.load(os.path.join(beta_dir,'subjects', f'{sub:03d}', 'SPM', 'spmT_S-N.nii'))
                    else:
                        file = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz'))
                    effect_data = transforms.mni152_to_fsaverage(file, fsavg_density='41k')
                    data_gii = []
                    for img in effect_data:
                        data_hemi = np.asarray(img.agg_data()).T
                        data_gii.append(data_hemi)
                    effect_array = np.hstack(data_gii)
                    all_data.append(effect_array)
            
                all_rsquared = []

                for i in range(len(all_data)):
                    sub_data = all_data[i]
                    other_data = [arr for j, arr in enumerate(all_data) if j != i]
                    other_data = np.stack(other_data)
                    mean_data = np.nanmean(other_data, axis=0)

                    mask_valid = ~np.isnan(sub_data)

                    X = mean_data[mask_valid].reshape(-1, 1)
                    y = sub_data[mask_valid]

                    
                    lin_reg = LinearRegression()
                    lin_reg.fit(X, y)
                    y_pred = lin_reg.predict(X)

                    if SCORE == 'determination':
                        rsquared = r2_score(y, y_pred)
                        all_rsquared.append(rsquared)

                    elif SCORE == 'corr':
                        r, _ = pearsonr(y, y_pred) #equivalent to correlating y and X
                        corr_sq = r**2
                        all_rsquared.append(corr_sq)

            # Save results
            with open(os.path.join(output_dir,f'{task}_{latent_var}_all_predict_from_beta_cv_r2_{SCORE}.pickle'), "wb") as fp:   
                pickle.dump(all_rsquared, fp)

            expl_variance = np.mean(all_rsquared)
            sem_r2 = sem(all_rsquared)
            outfile.write(f'{latent_var}: {expl_variance}, sem: {sem_r2}\n')

            outfile.write('\n\n')

if COMP_NULL:
    df = pd.DataFrame(index=params.tasks, columns=params.latent_vars)
    results = []  # store results for correction later

    # collect all p-values
    p_values = []

    for task in params.tasks:
        for latent_var in params.latent_vars:
            # Load empirical and null data
            emp = np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_cv_r2_{MODEL_TYPE}_{SCORE}.pickle'), allow_pickle=True)
            null = np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2_{SCORE}.pickle'), allow_pickle=True)
            null_df = pd.DataFrame(null)
            null_means = null_df.mean().tolist()

            # Paired t-test
            ttest = ttest_rel(null_means, emp, alternative='less')

            # Store results
            results.append({
                'task': task,
                'latent_var': latent_var,
                't_value': ttest.statistic,
                'p_value': ttest.pvalue,
                'df': len(null_means) - 1
            })
            p_values.append(ttest.pvalue)

    # Apply FDR correction across all tests
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    # Add corrected p-values and rejection flags back into results
    for i, r in enumerate(results):
        r['p_value_fdr'] = pvals_corrected[i]
        r['significant_fdr'] = reject[i]
        df.loc[r['task'], r['latent_var']] = r['p_value_fdr']

    # Save all results to file
    with open(os.path.join(output_dir, f'compare_emp_null_cv_{SCORE}.txt'), "a") as outfile:
        for r in results:
            outfile.write(f"{r['task']} and {r['latent_var']}:\n")
            outfile.write(f"t-value: {r['t_value']}\n")
            outfile.write(f"p-value: {r['p_value']}\n")
            outfile.write(f"FDR-corrected p-value: {r['p_value_fdr']}\n")
            outfile.write(f"Significant (FDR<0.05): {r['significant_fdr']}\n")
            outfile.write(f"df: {r['df']}\n\n")

    # Optionally save FDR-corrected table
    df.to_csv(os.path.join(output_dir, f'fdr_corrected_pvalues_{SCORE}.csv'))


if COMPARE_EXPL_VAR:
    n_perm = 10000
    rng = np.random.default_rng(seed=123)

    # Function to compute group ratio
    def group_ratio(rec_var, expl_var):
        return np.mean(rec_var) / np.mean(expl_var)

    # Load reference (lanA)
    expl_var_lanA = np.array(np.load(os.path.join(output_dir, f'lanA_S-N_all_predict_from_beta_cv_r2_{SCORE}.pickle'), allow_pickle=True))
    rec_var_lanA  = np.array(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_cv_r2_{MODEL_TYPE}_{SCORE}.pickle'), allow_pickle=True))
    ratio_lanA = group_ratio(rec_var_lanA, expl_var_lanA)

    # Store results for later FDR correction
    results = []

    for task in params.tasks:
        for latent_var in params.latent_vars:
            print(f"Running permutation for {task} / {latent_var}...")

            # Load task data
            expl_var_task = np.array(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_predict_from_beta_cv_r2_{SCORE}.pickle'), allow_pickle=True))
            rec_var_task  = np.array(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_cv_r2_{MODEL_TYPE}_{SCORE}.pickle'), allow_pickle=True))

            ratio_task = group_ratio(rec_var_task, expl_var_task)
            obs_diff = ratio_lanA - ratio_task

            # --- Permutation test ---
            combined_rec = np.concatenate([rec_var_lanA, rec_var_task])
            combined_expl = np.concatenate([expl_var_lanA, expl_var_task])
            n_lanA = len(rec_var_lanA)

            perm_diffs = np.empty(n_perm)
            for i in range(n_perm):
                perm_indices = rng.permutation(len(combined_rec))
                rec_perm_A = combined_rec[perm_indices[:n_lanA]]
                rec_perm_B = combined_rec[perm_indices[n_lanA:]]
                expl_perm_A = combined_expl[perm_indices[:n_lanA]]
                expl_perm_B = combined_expl[perm_indices[n_lanA:]]
                perm_diffs[i] = group_ratio(rec_perm_A, expl_perm_A) - group_ratio(rec_perm_B, expl_perm_B)

            # Two-sided p-value
            p_val = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
            ci_lower, ci_upper = np.percentile(perm_diffs, [2.5, 97.5])

            results.append({
                "task": task,
                "latent_var": latent_var,
                "ratio_lanA": ratio_lanA,
                "ratio_task": ratio_task,
                "obs_diff": obs_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_val": p_val
            })

    # --- FDR correction across all comparisons ---
    p_values = [r["p_val"] for r in results]
    reject, pvals_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Assign corrected p-values
    for i, r in enumerate(results):
        r["p_val_fdr"] = pvals_fdr[i]
        r["reject_null"] = reject[i]

    # --- Write output ---
    outfile_path = os.path.join(output_dir, f'compare_explained_variance_cv_group_ratio_{SCORE}_permtest_fdr.txt')
    with open(outfile_path, "w") as outfile:
        for r in results:
            outfile.write(f"{r['task']} and {r['latent_var']}:\n")
            outfile.write(f'Group Ratio lanA: {r["ratio_lanA"]:.4f}\n')
            outfile.write(f'Group Ratio {r["task"]}_{r["latent_var"]}: {r["ratio_task"]:.4f}\n')
            outfile.write(f'Diff in Group Ratios (lanA - task): {r["obs_diff"]:.4f}\n')
            outfile.write(f'95% Permutation CI: [{r["ci_lower"]:.4f}, {r["ci_upper"]:.4f}]\n')
            outfile.write(f'Permutation p-value: {r["p_val"]:.4f}\n')
            outfile.write(f'FDR-corrected p-value: {r["p_val_fdr"]:.4f}\n')
            outfile.write(f'Significant after FDR (q<0.05): {r["reject_null"]}\n\n')



def plot_variance_explained(params, model_type="linear", score="determination", comparison_latent="S-N"):
    """
    Plot variance explained and null model for each latent variable and task.
    """
    stats = {}
    for latent_var in params.latent_vars:
        for task in params.tasks:
            emp = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"{task}_{latent_var}_all_regression_cv_r2_{score}.pickle"), allow_pickle=True))
            null = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"{task}_{latent_var}_all_regression_null_cv_r2.pickle"), allow_pickle=True))

            stats[(latent_var, task)] = {
                "mean_emp": np.nanmean(emp),
                "sem_emp": np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp))),
                "mean_null": np.nanmean(null),
            }

    # Add the language comparison
    emp = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"lanA_S-N_all_regression_cv_r2_{score}.pickle"), allow_pickle=True))
    null = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"lanA_S-N_all_regression_null_cv_r2.pickle"), allow_pickle=True))
    stats[("language", "lanA")] = {
        "mean_emp": np.nanmean(emp),
        "sem_emp": np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp))),
        "mean_null": np.nanmean(null),
    }

    all_latents = params.latent_vars + [comparison_latent]

    # --- compute positions ---
    n_tasks = len(params.tasks)
    group_width, inner_pad, cursor = 0.7, 0.3, 0
    slot_width = group_width / n_tasks
    bar_width = slot_width * (1 - inner_pad)
    x_positions, group_centers = [], []

    for latent in params.latent_vars:
        group_left = cursor
        group_center = group_left + group_width / 2
        group_centers.append(group_center)
        for ti, task in enumerate(params.tasks):
            x_positions.append((latent, task, group_left + (ti + 0.5) * slot_width))
        cursor += group_width + 0.3

    bar_center = cursor + bar_width / 2
    group_centers.append(bar_center)
    x_positions.append(("language", "lanA", bar_center))

    # --- plotting ---
    fig, ax = plt.subplots() 
    task_colors = ["#B32E25", "#E06D38", "#4460AB", "#80C0F7"]

    for latent, task, x in x_positions:
        s = stats[(latent, task)]
        base_color = "gray" if latent == "language" else task_colors[params.tasks.index(task)]

        if task == "NAConf":
            ax.bar(
                x, s["mean_emp"], width=bar_width,
                yerr=s["sem_emp"],
                color=base_color, edgecolor="black", linewidth=1, alpha=0.8
            )
        else:
            # split bar
            ax.bar(
                x, s["mean_null"], width=bar_width, color=base_color,
                edgecolor="black", linewidth=1, alpha=0.3
            )
            ax.bar(
                x, s["mean_emp"] - s["mean_null"], bottom=s["mean_null"],
                width=bar_width, yerr=s["sem_emp"],
                color=base_color, edgecolor="black", linewidth=1, alpha=0.8
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_latents)
    ax.set_ylabel("R² (CV)")
    ax.set_ylim(0, 0.06)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.margins(x=0.06)

    # Legend remapping
    mapped_task_labels = [
        params.study_mapping.get(t_orig, t_orig) for t_orig in params.tasks
    ]

    comparison_patch = Patch(facecolor="gray", edgecolor="black", label="Language network")
    task_handles = [
        Patch(facecolor=task_colors[i], edgecolor="black", label=mapped_task_labels[i])
        for i in range(len(params.tasks))
    ]

    ax.legend(
        handles=task_handles + [comparison_patch],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2
    )
    return fig, ax


def plot_explained_variance_ratio(params, model_type="lin", score="determination", comparison_latent="S-N", n_boot=10000):
    """
    Plot the explainable variance ratio (empirical / predicted) with bootstrap CIs.
    Uses the currently active plotting style.
    """
    rng = np.random.default_rng(seed=124)

    def bootstrap_group_ratio(rec_var, expl_var, n_boot=10000, rng=None):
        n = len(rec_var)
        return np.array([
            np.mean(rng.choice(rec_var, size=n, replace=True)) / 
            np.mean(rng.choice(expl_var, size=n, replace=True))
            for _ in range(n_boot)
        ])

    stats = {}
    for latent_var in params.latent_vars:
        for task in params.tasks:
            rec_path = os.path.join(paths.home_dir, 'variance_explained', f"{task}_{latent_var}_all_regression_cv_r2_{score}.pickle")
            expl_path = os.path.join(paths.home_dir, 'variance_explained', f"{task}_{latent_var}_all_predict_from_beta_cv_r2_{score}.pickle")
            rec_var, expl_var = np.asarray(np.load(rec_path, allow_pickle=True)), np.asarray(np.load(expl_path, allow_pickle=True))
            mask = (~np.isnan(rec_var)) & (~np.isnan(expl_var))
            rec_var, expl_var = rec_var[mask], expl_var[mask]
            boot_ratios = bootstrap_group_ratio(rec_var, expl_var, rng=rng)
            mean_ratio = np.mean(boot_ratios)
            ci_lower, ci_upper = np.percentile(boot_ratios, [2.5, 97.5])
            sem_ratio = (ci_upper - ci_lower) / 3.92
            stats[(latent_var, task)] = {"mean_ratio": mean_ratio, "sem_ratio": sem_ratio}

    # Special case: language
    rec_var = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"lanA_S-N_all_regression_cv_r2_{score}.pickle"), allow_pickle=True))
    expl_var = np.asarray(np.load(os.path.join(paths.home_dir, 'variance_explained', f"lanA_S-N_all_predict_from_beta_cv_r2_{score}.pickle"), allow_pickle=True))
    mask = (~np.isnan(rec_var)) & (~np.isnan(expl_var))
    rec_var, expl_var = rec_var[mask], expl_var[mask]
    boot_ratios = bootstrap_group_ratio(rec_var, expl_var, rng=rng)
    mean_ratio = np.mean(boot_ratios)
    ci_lower, ci_upper = np.percentile(boot_ratios, [2.5, 97.5])
    sem_ratio = (ci_upper - ci_lower) / 3.92
    stats[("language", "lanA")] = {"mean_ratio": mean_ratio, "sem_ratio": sem_ratio}

    all_latents = params.latent_vars + [comparison_latent]
    n_tasks = len(params.tasks)

    # --- positions ---
    group_width, inner_pad, cursor = 0.7, 0.3, 0
    slot_width = group_width / n_tasks
    bar_width = slot_width * (1 - inner_pad)
    x_positions, group_centers = [], []

    for latent in params.latent_vars:
        group_left = cursor
        group_centers.append(group_left + group_width / 2)
        for ti, task in enumerate(params.tasks):
            x_positions.append((latent, task, group_left + (ti + 0.5) * slot_width))
        cursor += group_width + 0.3

    bar_center = cursor + bar_width / 2
    group_centers.append(bar_center)
    x_positions.append(("language", "lanA", bar_center))

    # --- plotting ---
    fig, ax = plt.subplots() 
    task_colors = ["#B32E25", "#E06D38", "#4460AB", "#80C0F7"]

    for latent, task, x in x_positions:
        s = stats[(latent, task)]
        color = "gray" if latent == "language" else task_colors[params.tasks.index(task)]
        ax.bar(
            x, s["mean_ratio"], width=bar_width,
            yerr=s["sem_ratio"], 
            color=color, edgecolor="black", linewidth=1, alpha=0.8
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_latents)
    ax.set_ylabel("Explainable variance ratio")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.margins(x=0.06)
    ax.set_yticks(np.linspace(0, 0.4, 5))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 0.4, 5)])

    mapped_task_labels = [
        next((params.study_mapping[t] for t in params.study_mapping if t in t_orig), t_orig)
        for t_orig in params.tasks
    ]

    # Legend
    comparison_patch = Patch(facecolor="gray", edgecolor="black", label="Language network")
    task_handles = [
        Patch(facecolor=task_colors[i], edgecolor="black", label=mapped_task_labels[i])
        for i in range(len(params.tasks))
    ]

    ax.legend(
        handles=task_handles + [comparison_patch],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2 
    )

    plt.tight_layout()
    return fig, ax

mf.set_publication_style(font_size=8)

if PLOT_VAR_EXPLAINED:
    fig, ax = plot_variance_explained(params)
    mf.save_figure(fig, output_dir, f"barplot_explained_variance_mean_{SCORE}")

if PLOT_VAR_EXPLAINED_RATIO:
    fig, ax = plot_explained_variance_ratio(params)
    mf.save_figure(fig, output_dir, f"barplot_explained_variance_ratio_mean_{SCORE}")




