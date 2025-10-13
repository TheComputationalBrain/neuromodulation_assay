import os
import glob
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
from neuromaps import transforms
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore, sem, ttest_rel
import main_funcs as mf
from params_and_paths_analysis import Paths, Params, Receptors
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nilearn import datasets


FROM_BETA = False
FROM_RECEPTOR = True
COMP_NULL = False
COMPARE_LANG_LEARN = False
COMPARE_EXPL_VAR = False
PLOT_VAR_EXPLAINED = False
PLOT_REL_VAR_EXPLAINED = False
PLOT_PERCENT_VAR_EXPLAINED = False

model_type = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'

ON_SURFACE = True
WEIGHT_MODE = "subjects"   # options: "voxels" or "subjects" 


if ON_SURFACE:
    proj = '_on_surf'
else:
    proj = ''

suffix = ''

paths = Paths()
params = Params()
rec = Receptors()

output_dir = os.path.join(paths.home_dir, 'variance_explained')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

comparison_latent = 'language'
comparison_task = 'lanA'

fsavg = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

if FROM_BETA:
    for task in params.tasks: 

        if task == 'Explore':
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
        elif task == 'lanA':
            beta_dir = os.path.join(paths.home_dir, paths.fmri_dir['lanA'])
        else:
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        subject_paths = params.home_dir if task == "lanA" else params.root_dir
        subjects = mf.get_subjects(task, os.path.join(subject_paths, paths.fmri_dir[task]))
        subjects = [subj for subj in subjects if subj not in params.ignore[task]] 

        with open(os.path.join(output_dir,f'predict_from_beta{proj}.txt'), "a") as outfile:
            outfile.write(f'{task}: variance explained in analysis:\n\n')
            
            for latent_var in params.latent_vars:
                all_data = []

                #load all data
                for sub in subjects:
                    if ON_SURFACE:
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
                    else:
                        sub_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{params.mask}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()
                        all_data.append(sub_data)

                all_rsquared = []

                for i in range(len(all_data)):
                    sub_data = all_data[i]
                    other_data = [arr for j, arr in enumerate(all_data) if j != i]
                    other_data = np.stack(other_data)
                    mean_data = np.nanmean(other_data, axis=0)

                    for i in range(len(all_data)):
                        sub_data = all_data[i]
                        other_data = [arr for j, arr in enumerate(all_data) if j != i]
                        other_data = np.stack(other_data)
                        mean_data = np.nanmean(other_data, axis=0)

                        mask_valid = ~np.logical_or(np.isnan(sub_data), np.isclose(sub_data, 0)).flatten() if ON_SURFACE else ~np.isnan(sub_data)

                        X = mean_data[mask_valid].reshape(-1, 1)
                        y = sub_data[mask_valid]

                        lin_reg = LinearRegression()
                        lin_reg.fit(X, y)
                        rsquared = lin_reg.score(X, y)
                        all_rsquared.append(rsquared)

                # Save results
                with open(os.path.join(output_dir,f'{task}_{latent_var}_all_predict_from_beta_cv_r2{proj}{suffix}.pickle'), "wb") as fp:   
                    pickle.dump(all_rsquared, fp)

                expl_variance = np.mean(all_rsquared)
                sem_r2 = sem(all_rsquared)
                outfile.write(f'{latent_var}: {expl_variance}, sem: {sem_r2}\n')

            outfile.write('\n\n')


if FROM_RECEPTOR:
    df = pd.DataFrame(index=params.tasks, columns=params.latent_vars)
    for task in params.tasks: 
        if task == 'Explore':
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
        elif task == 'lanA':
            beta_dir = os.path.join(paths.home_dir, paths.fmri_dir['lanA'])
        else:
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

        subject_paths = params.home_dir if task == "lanA" else params.root_dir
        subjects = mf.get_subjects(task, os.path.join(subject_paths, paths.fmri_dir[task]))
        subjects = [subj for subj in subjects if subj not in params.ignore[task]] 

        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""    

        if ON_SURFACE: 
            receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
            receptor_density =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
            params.mask = params.mask 
        else:                                            
            if params.parcelated:
                receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
                params.mask = params.mask + '_' + params.mask_details 
                receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True), nan_policy='omit') 
                text = 'by region'
            else:
                receptor_dir = os.path.join(paths.home_dir, 'receptors', 'PET2') #vertex level analyis can only be run on PET data densities 
                receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
                params.mask = params.mask 
                text = 'by voxel'

            if rec.source == 'autorad_zilles44':
                #autoradiography dataset is only one hemisphere 
                receptor_density = np.concatenate((receptor_density, receptor_density))

        with open(os.path.join(output_dir,f'predict_from_receptor{proj}.txt'), "a") as outfile:
            outfile.write(f'{task}: variance explained in analysis with {rec.source} as predictor:\n\n')
            n_features = receptor_density.shape[1]
            for latent_var in params.latent_vars:
                print(f'latent variable: {latent_var}')
                r2_scores = [] 
                if ON_SURFACE:
                    if task == 'lanA':
                        fmri_files = []
                        for subj in subjects:
                            subj_id = f"{subj:03d}"  
                            pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
                            fmri_files.extend(glob.glob(pattern))
                    else:
                        fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz')))
                        fmri_files = []
                        for file in fmri_files_all:
                            basename = os.path.basename(file)
                            subj_str = basename.split('_')[0]  # 'sub-XX'
                            subj_id = int(subj_str.split('-')[1])  # XX as integer
                            if subj_id in subjects:
                                fmri_files.append(file)
                    fmri_activity = []
                    for file in fmri_files:
                        data_vol = nib.load(file)
                        effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
                        data_gii = []
                        for img in effect_data:
                            data_hemi = img.agg_data()
                            data_hemi = np.asarray(data_hemi).T
                            data_gii += [data_hemi]
                            effect_array = np.hstack(data_gii)    
                        fmri_activity.append(effect_array) 
                    valid_counts = []
                    for idx, arr in enumerate(fmri_activity):
                        mask_valid = ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
                        n_valid = np.sum(mask_valid)
                        valid_counts.append(n_valid)
                else:    
                    fmri_files = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size{add_info}.pickle')))
                    fmri_activity = []
                    for file in fmri_files:
                        with open(file, 'rb') as f:
                            fmri_activity.append(pickle.load(f))  
                
                for i in range(len(subjects)):
                    X_train, y_train, weights = [], [], []

                    for j in range(len(subjects)):
                        if j == i:
                            continue
                        mask_j = ~np.logical_or(np.isnan(fmri_activity[j]), np.isclose(fmri_activity[j], 0)).flatten()
                        valid_data = fmri_activity[j].flatten()[mask_j]
                        Xj = receptor_density[mask_j]
                        yj = zscore(valid_data)

                        # weight = 1 / n_voxels for this subject
                        subj_weight = np.ones(len(yj)) / len(yj)

                        X_train.append(Xj)
                        y_train.append(yj)
                        if WEIGHT_MODE == "subjects":
                            subj_weight = np.ones(len(yj)) / len(yj)   # scale subject to weight=1
                            weights.append(subj_weight)

                    # Concatenate everything
                    X_train = np.concatenate(X_train)
                    y_train = np.concatenate(y_train)
                    if WEIGHT_MODE == "subjects":
                        weights = np.concatenate(weights)
                    else:
                        weights = None

                    # Test subject
                    mask_i = ~np.logical_or(np.isnan(fmri_activity[i]), np.isclose(fmri_activity[i], 0)).flatten()
                    X_test = receptor_density[mask_i]
                    y_test = zscore(fmri_activity[i].flatten()[mask_i])

                    # Fit with sample weights
                    model = LinearRegression()
                    model.fit(X_train, y_train, sample_weight=weights)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    r2_scores.append(r2)

                average_r2 = np.mean(r2_scores)
                sem_r2 = sem(r2_scores)
                outfile.write(f'{latent_var}: {average_r2}, sem: {sem_r2}\n')
                df.loc[task, latent_var] = average_r2

                if model_type == 'linear':
                    with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2{proj}{suffix}.pickle'), "wb") as fp:   
                        pickle.dump(r2_scores, fp)
                else:
                    with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2{proj}_{model_type}.pickle'), "wb") as fp:   
                        pickle.dump(r2_scores, fp)

            outfile.write('\n\n')
    if model_type == 'linear':
        df.to_csv(os.path.join(output_dir,f'overview_regression_cv{proj}_weighted.csv'))
    else:
        df.to_csv(os.path.join(output_dir,f'overview_regression_cv{proj}_{model_type}_weighted.csv'))

if COMP_NULL:
    #latent_vars = ['S-N'] 
    df = pd.DataFrame(index=params.tasks, columns=params.latent_vars)

    with open(os.path.join(output_dir,f'compare_emp_null_cv_{proj}_{model_type}.txt'), "a") as outfile:
        for task in params.tasks: 
            for latent_var in params.latent_vars:
                outfile.write(f'{task} and {latent_var}:\n')
                #get the empirical r2 (on surf)
                emp = np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2_on_surf{suffix}.pickle'), allow_pickle=True)
                #get the null r2
                null = np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_null_cv_r2{suffix}.pickle'), allow_pickle=True)
                null_df = pd.DataFrame(null)
                null_means = null_df.mean().tolist()

                #summarize the null r2 as we did with the empirical r2 above (for table overview in paper)
                r2 = sum(null_means) / len(null_means)
                df.loc[task, latent_var] = r2

                #compare the left-out subject (mean) r2 via t-test 
                ttest = ttest_rel(null_means, emp, alternative='less') #welch test instead?
                outfile.write(f't-value: {ttest.statistic}\n') 
                outfile.write(f'p-value: {ttest.pvalue}\n')
                outfile.write(f'df: {ttest.df}\n')
                outfile.write(f'\n\n')

            df.to_csv(os.path.join(output_dir,f'overview_null_cv{proj}_{model_type}.csv'))

            outfile.write(f'\n\n')

if COMPARE_EXPL_VAR:
    # Number of bootstrap samples
    n_boot = 10000
    rng = np.random.default_rng(seed=124)

    # Function to bootstrap group-level ratio
    def bootstrap_group_ratio(rec_var, expl_var, n_boot=10000, rng=None):
        n = len(rec_var)
        return np.array([
            np.mean(rng.choice(rec_var, size=n, replace=True)) / 
            np.mean(rng.choice(expl_var, size=n, replace=True))
            for _ in range(n_boot)
        ])

    # Load reference (baseline) data
    expl_var = np.array(np.load(os.path.join(output_dir, f'lanA_S-N_all_predict_from_beta_cv_r2{proj}_2.pickle'), allow_pickle=True))
    rec_var = np.array(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_cv_r2{proj}_2.pickle'), allow_pickle=True))

    # Bootstrap the baseline group ratio
    boot_lanA = bootstrap_group_ratio(rec_var, expl_var, n_boot=n_boot, rng=rng)

    # Write results
    with open(os.path.join(output_dir, f'compare_explained_variance_cv_{proj}_group_ratio.txt'), "w") as outfile:
        for task in params.tasks:
            for latent_var in params.latent_vars:
                outfile.write(f'{task} and {latent_var}:\n')

                # Load task data
                expl_var = np.array(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_predict_from_beta_cv_r2{proj}.pickle'), allow_pickle=True))
                rec_var = np.array(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_cv_r2{proj}.pickle'), allow_pickle=True))

                # Bootstrap the task group ratio
                boot_task = bootstrap_group_ratio(rec_var, expl_var, n_boot=n_boot, rng=rng)

                # Compute the difference in bootstrapped group-level ratios
                boot_diff = boot_lanA - boot_task

                # Confidence interval
                ci_lower = np.percentile(boot_diff, 2.5)
                ci_upper = np.percentile(boot_diff, 97.5)

                # Two-sided bootstrap p-value
                p_val = min(np.mean(boot_diff <= 0), np.mean(boot_diff >= 0))

                # Write output
                outfile.write(f'Group Ratio lanA: {np.mean(boot_lanA):.4f}\n')
                outfile.write(f'Group Ratio {task}_{latent_var}: {np.mean(boot_task):.4f}\n')
                outfile.write(f'Diff in Group Ratios (lanA - task): {np.mean(boot_diff):.4f}\n')
                outfile.write(f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n')
                outfile.write(f'Bootstrap p-value: {p_val:.4f}\n\n')


#plot variance explained and null model 
if PLOT_VAR_EXPLAINED:
    stats = {}
    for latent_var in params.latent_vars:
        for task in params.tasks: 
            emp = np.asarray(np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2_on_surf.pickle'), allow_pickle=True))
            null = np.asarray(np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_null_cv_r2.pickle'), allow_pickle=True))
            mean_emp = np.nanmean(emp)
            sem_emp = np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp)))
            mean_null = np.nanmean(null)
            stats[(latent_var, task)] = {
                "mean_emp": np.nanmean(emp),
                "sem_emp": np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp))),
                "mean_null": np.nanmean(null)
            }
    
    #add the language network 
    emp = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_cv_r2_on_surf_2.pickle'), allow_pickle=True))
    null = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_null_cv_r2_2.pickle'), allow_pickle=True))
    stats[('language','lanA')] = {
        "mean_emp": np.nanmean(emp),
        "sem_emp": np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp))),
        "mean_null": np.nanmean(null)
    }

    all_latents = params.latent_vars + [comparison_latent]

    n_latents = len(params.latent_vars)
    n_tasks = len(params.tasks)

    x_positions = []
    group_centers = []

    # spacing and width parameters (same as before)
    group_width = 0.8
    inner_pad = 0.2
    slot_width = group_width / n_tasks
    bar_width  = slot_width * (1 - inner_pad)

    # start cursor at x=0
    cursor = 0

    # Process first two groups
    for latent in params.latent_vars:
        group_left = cursor
        group_right = group_left + group_width
        group_center = (group_left + group_right) / 2
        group_centers.append(group_center)
        for ti, task in enumerate(params.tasks):
            slot_center = group_left + (ti + 0.5) * slot_width
            x_positions.append((latent, task, slot_center))
        cursor = group_right + 0.3  # inter-group spacing: adjust 0.3 as needed

    # Process last group ('S-N') separately:
    bar_center = cursor + bar_width / 2
    group_centers.append(bar_center)
    x_positions.append(('language', 'lanA', bar_center))

    fig, ax = plt.subplots(figsize=(10, 5))
    task_colors = ["#B32E25", "#E06D38", "#4460AB", "#80C0F7"]

    for latent, task, x in x_positions:
        s = stats[(latent, task)]
        color = "gray" if latent == "language" else task_colors[params.tasks.index(task)]
        ax.bar(
            x,
            s["mean_emp"],
            width=bar_width,
            yerr=s["sem_emp"], capsize=3,
            color=color, edgecolor="black", linewidth=1, alpha=0.8
        )
        tick_half = bar_width * 0.6
        ax.hlines(s["mean_null"], x - tick_half, x + tick_half, colors="black", linewidth=2, linestyles='dashed')

        # nice margins around the outer groups
        ax.margins(x=0.06)

    # X-axis labels = latent variables
    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_latents)
    ax.set_ylabel("R² (CV)")
    ax.set_ylim(0,0.06)
    ax.axhline(0, color="gray", linewidth=0.8)

    # Legend for tasks
    comparison_patch = Patch(facecolor="gray", edgecolor="black", label="Language network")
    task_handles = [Patch(facecolor=task_colors[i], edgecolor="black", label=t) for i, t in enumerate(params.tasks)]
    null_line = Line2D([0], [0], color="black", lw=2, label="Null mean")
    ax.legend(handles=task_handles + [comparison_patch, null_line], 
            frameon=False,
            bbox_to_anchor=(1.05, 1),  
            loc='upper left')
    for text in ax.get_figure().findobj(plt.Text):
        text.set_fontsize(18)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'barplot_explained_variance.svg'), bbox_inches="tight")

# plot relative variance explained (empirical - null)
if PLOT_REL_VAR_EXPLAINED:
    stats = {}
    for latent_var in params.latent_vars:
        for task in params.tasks: 
            emp = np.asarray(np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2_on_surf.pickle'), allow_pickle=True))
            null = np.asarray(np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_null_cv_r2.pickle'), allow_pickle=True))
            
            mean_emp = np.nanmean(emp)
            sem_emp = np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp)))
            mean_null = np.nanmean(null)

            stats[(latent_var, task)] = {
                "mean_emp": mean_emp,
                "sem_emp": sem_emp,
                "mean_null": mean_null,
                "rel_emp": mean_emp - mean_null   # relative to null
            }
    
    # add the language network 
    emp = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_cv_r2_on_surf_2.pickle'), allow_pickle=True))
    null = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_null_cv_r2_2.pickle'), allow_pickle=True))
    stats[('language','lanA')] = {
        "mean_emp": np.nanmean(emp),
        "sem_emp": np.nanstd(emp, ddof=1) / np.sqrt(np.sum(~np.isnan(emp))),
        "mean_null": np.nanmean(null),
        "rel_emp": np.nanmean(emp) - np.nanmean(null)
    }

    all_latents = params.latent_vars + [comparison_latent]

    n_latents = len(params.latent_vars)
    n_tasks = len(params.tasks)

    x_positions = []
    group_centers = []

    # spacing and width parameters (same as before)
    group_width = 0.8
    inner_pad = 0.2
    slot_width = group_width / n_tasks
    bar_width  = slot_width * (1 - inner_pad)

    # start cursor at x=0
    cursor = 0

    # Process first two groups
    for latent in params.latent_vars:
        group_left = cursor
        group_right = group_left + group_width
        group_center = (group_left + group_right) / 2
        group_centers.append(group_center)
        for ti, task in enumerate(params.tasks):
            slot_center = group_left + (ti + 0.5) * slot_width
            x_positions.append((latent, task, slot_center))
        cursor = group_right + 0.3  # inter-group spacing

    # Process last group ('S-N') separately:
    bar_center = cursor + bar_width / 2
    group_centers.append(bar_center)
    x_positions.append(('language', 'lanA', bar_center))

    fig, ax = plt.subplots(figsize=(10, 5))
    task_colors = ["#B32E25", "#E06D38", "#4460AB", "#80C0F7"]

    for latent, task, x in x_positions:
        s = stats[(latent, task)]
        color = "gray" if latent == "language" else task_colors[params.tasks.index(task)]
        ax.bar(
            x,
            s["rel_emp"],   # plot relative explained variance
            width=bar_width,
            yerr=s["sem_emp"], capsize=3,
            color=color, edgecolor="black", linewidth=1, alpha=0.8
        )

        # nice margins around the outer groups
        ax.margins(x=0.06)

    # X-axis labels = latent variables
    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_latents)
    ax.set_ylabel("Relative R2 (emp - null)")
    ax.axhline(0, color="gray", linewidth=0.8)

    # Legend for tasks
    comparison_patch = Patch(facecolor="gray", edgecolor="black", label="Language network")
    task_handles = [Patch(facecolor=task_colors[i], edgecolor="black", label=t) for i, t in enumerate(params.tasks)]
    ax.legend(handles=task_handles + [comparison_patch], 
            frameon=False,
            bbox_to_anchor=(1.05, 1),  
            loc='upper left')

    for text in ax.get_figure().findobj(plt.Text):
        text.set_fontsize(18)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'barplot_relative_variance_explained.svg'), bbox_inches="tight")


if PLOT_PERCENT_VAR_EXPLAINED:
    n_boot = 10000
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
            rec_var = np.asarray(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_cv_r2{proj}.pickle'), allow_pickle=True))
            expl_var = np.asarray(np.load(os.path.join(output_dir, f'{task}_{latent_var}_all_predict_from_beta_cv_r2{proj}.pickle'), allow_pickle=True))

            #! remove NaNs and clip negatives if needed
            mask = (~np.isnan(rec_var)) & (~np.isnan(expl_var)) & (expl_var > 0)
            rec_var = rec_var[mask]
            expl_var = expl_var[mask]

            # Bootstrap group-level ratio
            boot_ratios = bootstrap_group_ratio(rec_var, expl_var, rng=rng)

            mean_ratio = np.mean(boot_ratios)
            ci_lower, ci_upper = np.percentile(boot_ratios, [2.5, 97.5])
            sem_ratio = (ci_upper - ci_lower) / 3.92  # Approximate SEM from 95% CI

            stats[(latent_var, task)] = {
                "mean_ratio": mean_ratio,
                "sem_ratio": sem_ratio,
                "ci": (ci_lower, ci_upper)
            }

    # Special case for S-N (lanA)
    rec_var = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_regression_cv_r2{proj}_2.pickle'), allow_pickle=True))
    expl_var = np.asarray(np.load(os.path.join(output_dir, f'lanA_S-N_all_predict_from_beta_cv_r2{proj}_2.pickle'), allow_pickle=True))

    mask = (~np.isnan(rec_var)) & (~np.isnan(expl_var)) & (expl_var > 0)
    rec_var = rec_var[mask]
    expl_var = expl_var[mask]

    boot_ratios = bootstrap_group_ratio(rec_var, expl_var, rng=rng)

    mean_ratio = np.mean(boot_ratios)
    ci_lower, ci_upper = np.percentile(boot_ratios, [2.5, 97.5])
    sem_ratio = (ci_upper - ci_lower) / 3.92

    stats[('language', 'lanA')] = {
        "mean_ratio": mean_ratio,
        "sem_ratio": sem_ratio,
        "ci": (ci_lower, ci_upper)
    }

    all_latents = params.latent_vars + [comparison_latent]

    n_latents = len(params.latent_vars)
    n_tasks = len(params.tasks)

    x_positions = []
    group_centers = []

    # spacing and width parameters (same as before)
    group_width = 0.8
    inner_pad = 0.2
    slot_width = group_width / n_tasks
    bar_width  = slot_width * (1 - inner_pad)

    # start cursor at x=0
    cursor = 0

    # Process first two groups
    for latent in params.latent_vars:
        group_left = cursor
        group_right = group_left + group_width
        group_center = (group_left + group_right) / 2
        group_centers.append(group_center)
        for ti, task in enumerate(params.tasks):
            slot_center = group_left + (ti + 0.5) * slot_width
            x_positions.append((latent, task, slot_center))
        cursor = group_right + 0.3  # inter-group spacing: adjust 0.3 as needed

    # Process last group ('S-N') separately:
    bar_center = cursor + bar_width / 2
    group_centers.append(bar_center)
    x_positions.append(('language', 'lanA', bar_center))

    fig, ax = plt.subplots(figsize=(10, 5))
    task_colors = ["#B32E25", "#E06D38", "#4460AB", "#80C0F7"]

    for latent, task, x in x_positions:
        s = stats[(latent, task)]
        color = "gray" if latent == "language" else task_colors[tasks.index(task)]
        ax.bar(
            x,
            s["mean_ratio"], 
            width=bar_width,
            yerr=s["sem_ratio"], capsize=3, 
            color=color, edgecolor="black", linewidth=1, alpha=0.8
        )
        tick_half = bar_width * 0.6
        # nice margins around the outer groups
        ax.margins(x=0.06)
    # X-axis labels = latent variables
    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_latents)
    ax.set_ylabel("% Explainable Variance Explained")
    ax.axhline(0, color="gray", linewidth=0.8)
    custom_ticks = np.linspace(0, 0.4, 5)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in custom_ticks])

    # Legend for tasks
    comparison_patch = Patch(facecolor="gray", edgecolor="black", label="Language network")
    task_handles = [Patch(facecolor=task_colors[i], edgecolor="black", label=t) for i, t in enumerate(params.tasks)]
    ax.legend(
        handles=task_handles + [comparison_patch],
        frameon=False,
        bbox_to_anchor=(1.05, 1),  
        loc='upper left'
    )
    for text in ax.get_figure().findobj(plt.Text):
        text.set_fontsize(18)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'barplot_percent_explained_variance.svg'), bbox_inches="tight")




