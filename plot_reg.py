import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from params_and_paths import Paths, Params, Receptors

PLOT_COEFS = True
PLOT_DOMINANCE = False
FROM_OLS = False
PARCELATED = True

paths = Paths()
params = Params()
rec = Receptors()

if rec.source == 'PET':
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.dopamine, rec.cannabinnoid]
elif rec.source  == 'autorad_zilles44':
    receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.glutamate, rec.gaba, rec.dopamine]
receptor_class = [rec.exc,rec.inh]

if FROM_OLS:
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'OLS')
else: 
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level')

if PARCELATED:
    mask_comb = params.mask + '_' + params.mask_details
else:
    mask_comb = params.mask 

output_dir = os.path.join(beta_dir, 'regressions', rec.source)
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if PLOT_COEFS:
    #plot regression coefficients
    for y_name in params.io_regs:
        fname = f'{y_name}_{mask_comb}_regression_results_bysubject_all.csv'
        results_df = pd.read_csv(os.path.join(output_dir, fname))

        #t-test
        t_test_results = []
        p_values = []
        for receptor in rec.receptor_names:
            t_stat, p_value = ttest_1samp(results_df[receptor], 0)
            t_test_results.append({'receptor': receptor, 't_stat': t_stat, 'p_value': p_value})
            p_values.append(p_value)

        _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        significant_receptors = []
        for receptor, p_value_corrected in zip(rec.receptor_names, p_values_corrected):
            if p_value_corrected < 0.05:
                significant_receptors.append(receptor)

        ttest_df = pd.DataFrame(t_test_results)

        mean_R2 = results_df['R2'].mean()
        #std_R2 = results_df['R2'].std()
        mean_R2_adj = results_df['adjusted_R2'].mean()
        mean_BIC = results_df['BIC'].mean()


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
                ax.scatter(i, results_df[receptor].median(), color='red', zorder=5)  # Mark the median with a red dot

        ax.set_xticks(np.arange(len(ordered_receptors)))
        ax.set_xticklabels(ordered_receptors, rotation=90)
        for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
            group_idx = receptor_to_group.get(receptor, -1)
            label.set_color(base_colors[group_idx])
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Coef Values')
        ax.set_title(f'{mask_comb}: regression coefficients for {y_name} (FDR corrected)')

        # Add mean and standard deviation of R² to the plot
        textstr = f'Mean R²: {mean_R2:.2f}\nAdjusted mean R²: {mean_R2_adj:.2f}\nmean BIC: {mean_BIC:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        plt.tight_layout()
        plt.show()
        
        fname = f'{y_name}_{mask_comb}_all_reg_coef.png'
        fig_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)


#plot dominance analysis 

if PLOT_DOMINANCE:
    for y_name in params.io_regs:

        try:
            results_df = pd.read_pickle(os.path.join(output_dir, f'{y_name}_{mask_comb}_dominance_allsubj.pickle'))
        except FileNotFoundError:
            print(f"File not found for {y_name}, skipping...") #for now I only have the dominance data (computationally intensive) for surprise and confidence 
            continue

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

        # Standardize the data by total model fit
        standardized_df = results_df[ordered_receptors].div(results_df[ordered_receptors].sum(axis=1), axis=0)

        sem_data = standardized_df[ordered_receptors].sem() 
        bar_data = standardized_df[ordered_receptors].mean()
        bar_data = bar_data.reset_index()
        bar_data.columns = ['receptor', 'mean_value']
        bar_data['sem'] = sem_data.values

        bars = ax.bar(bar_data['receptor'], bar_data['mean_value'], yerr=bar_data['sem'],
                    color=[color['face'] for color in colors], edgecolor=[color['edge'] for color in colors], capsize=5)

        # Manually add hatch patterns
        for i, (receptor, color) in enumerate(zip(ordered_receptors, colors)):
            if receptor not in receptor_class[0] and receptor not in receptor_class[1]:
                bars[i].set_hatch('//')
            
        ax.set_xticks(np.arange(len(ordered_receptors)))
        ax.set_xticklabels(ordered_receptors, rotation=90)
        for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
            group_idx = receptor_to_group.get(receptor, -1)
            label.set_color(base_colors[group_idx])
        ax.set_xlabel('Receptor/Transporter')
        ax.set_ylabel('contribution (%)')
        #ax.set_title(f'{MASK_NAME}: dominance analysis for {y_name}')
        plt.tight_layout()
        
        fname = f'{y_name}_{mask_comb}_dominance.png'
        fig_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)

