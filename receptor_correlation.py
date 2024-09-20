import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, zscore, pearsonr
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors

RUN_CORR = True
PLOT_CORR = True

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

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
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) #vertex level analyis can only be run on PET data densities 
    output_dir = os.path.join(beta_dir, 'regressions', rec.source)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
    mask_comb = params.mask 

#TDODO: run on just one hemisphere instead
if rec.source == 'autorad_zilles44':
    #autoradiography dataset is only one hemisphere 
    receptor_density = np.concatenate((receptor_density, receptor_density))

if RUN_CORR:
    for latent_var in params.latent_vars:
        results_df = pd.DataFrame(columns=rec.receptor_names)

        for sub in subjects: 
            cors = np.zeros((len(rec.receptor_names),))
            y_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size.pickle'), allow_pickle=True).flatten()

            for indx,receptor in enumerate(rec.receptor_names):
                receptor = receptor_density[:,indx]
                if params.parcelated:
                    non_nan_indices = ~np.isnan(receptor)
                    X = receptor[non_nan_indices] #manual assignment of autorad data means that some regions are empty
                    y = y_data[non_nan_indices]
                else:
                    non_nan_indices = ~np.isnan(y_data)
                    X = receptor[non_nan_indices] #non parcelated data might contain a few NaNs from voxels with constant activation 
                    y = y_data[non_nan_indices]

                cors[indx] = pearsonr(X, y).correlation
                
            #results by functional activity across participants 
            results = pd.DataFrame([cors], columns = rec.receptor_names)
            results_df = pd.concat([results_df,results], ignore_index=True, sort=False)

        fname = f'{latent_var}_{mask_comb}_correlation_results_bysubject.csv'
        results_df.to_csv(os.path.join(output_dir, fname), index=False)  

if PLOT_CORR:
# plot correlation results in Boxplot
    if rec.source in ['PET', 'PET2']:
            receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.dopamine, rec.cannabinnoid]
    elif rec.source  == 'autorad_zilles44':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.glutamate, rec.gaba, rec.dopamine]
    elif rec.source == 'AHBA':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.dopamine]
    receptor_class = [rec.exc,rec.inh]

    plot_dir = os.path.join(output_dir, 'plots_corr')
    if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    plt.rcParams.update({'font.size': 16})

    #by latent variable
    for latent_var in params.latent_vars:
        fname = f'{latent_var}_{mask_comb}_correlation_results_bysubject.csv'
        results_df = pd.read_csv(os.path.join(output_dir, fname))

        #t-test
        t_test_results = []
        significant_receptors = []
        for receptor in rec.receptor_names:
            t_stat, p_value = ttest_1samp(results_df[receptor], 0)
            t_test_results.append({'receptor': receptor, 't_stat': t_stat, 'p_value': p_value})
            if p_value < 0.05:
                significant_receptors.append(receptor)

        ttest_df = pd.DataFrame(t_test_results)

        ##### group boxplot by neuromodulator and ex/inhb #####
        receptor_to_group = {}
        for group_idx, group in enumerate(receptor_groups):
            for receptor in group:
                receptor_to_group[receptor] = group_idx
        
        ordered_receptors = [receptor for group in receptor_groups for receptor in group]

        receptor_to_class = {}
        for class_idx, class_group in enumerate(receptor_class):
            for receptor in class_group:
                receptor_to_class[receptor] = class_idx

        # Assign colors to each group
        base_colors = sns.color_palette('husl', len(receptor_groups))
        colors = []
        for receptor in ordered_receptors:
            group_idx = receptor_to_group.get(receptor, -1)
            class_type = receptor_to_class.get(receptor, -1)
            if class_type == 0:  # Excitatory
                color = sns.dark_palette(base_colors[group_idx], n_colors=3)[1]
                colors.append({'face': color, 'edge': color})
            elif class_type == 1:  # Inhibitory
                color = sns.light_palette(base_colors[group_idx], n_colors=3)[1]
                colors.append({'face': color, 'edge': color})
            else:
                face_color = sns.light_palette(base_colors[group_idx], n_colors=3)[0]
                edge_color = sns.dark_palette(base_colors[group_idx], n_colors=3)[2]
                colors.append({'face': face_color, 'edge': edge_color})

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (receptor, color) in enumerate(zip(ordered_receptors, colors)):
            sns.boxplot(data=results_df[[receptor]], ax=ax, color=color['face'], boxprops=dict(facecolor=color['face'], hatch='//', edgecolor=color['edge']), positions=[i])

        for i, receptor in enumerate(ordered_receptors):
            if receptor in significant_receptors:
                ax.scatter(i, results_df[receptor].median(), color='red', zorder=5)  

        ax.set_xticks(np.arange(len(ordered_receptors)))
        ax.set_xticklabels(ordered_receptors, rotation=90)
        for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
            group_idx = receptor_to_group.get(receptor, -1)
            label.set_color(base_colors[group_idx])
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Correlation')
        ax.set_title(f'{mask_comb}: individual correlations for {latent_var} (FDR corrected) with {rec.source}', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        fname = f'{latent_var}_{mask_comb}_all_corr.png'
        fig_dir = os.path.join(output_dir, 'plots_corr')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)