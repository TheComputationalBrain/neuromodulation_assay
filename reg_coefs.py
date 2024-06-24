import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from params_and_paths import *

n_reg = 'all' #all or core
DENSITY_BY_TRACER = True
FROM_OLS = True

y_names = np.array(['surprise','confidence', 'predictability', 'predictions'])

if n_reg == 'core':
    receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                                "D1", "D2", "DAT", "GABAa", "M1", "mGluR5",
                                "NET", "NMDA", "VAChT"])
    serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
    acetylcholine = ["A4B2", "M1", "VAChT"]
    noradrenaline = ["NET"]
    glutamate = ["mGluR5"]
    gaba = ["GABAa"]
    dopamine = ["D1", "D2", "DAT"]
    receptor_groups = [serotonin, acetylcholine, noradrenaline, glutamate, gaba, dopamine]
    exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
    inh = ['5HT1a', '5HT1b', 'D2', 'GABAa', 'H3']
    receptor_class = [exc,inh]
else:
    y_names = np.array(['surprise','confidence', 'predictability', 'predictions'])
    receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                            "MOR", "NET", "NMDA", "VAChT"])
    serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
    acetylcholine = ["A4B2", "M1", "VAChT"]
    noradrenaline = ["NET"]
    opioid = ["MOR"]
    glutamate = ["mGluR5"]
    histamine = ["H3"]
    gaba = ["GABAa"]
    dopamine = ["D1", "D2", "DAT"]
    cannabinnoid = ["CB1"]
    receptor_groups = [serotonin, acetylcholine, noradrenaline, opioid, glutamate, histamine, gaba, dopamine, cannabinnoid]
    exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
    inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']
    receptor_class = [exc,inh]

if FROM_OLS:
    beta_dir  = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level', 'OLS')
else: 
    beta_dir  = os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level')
                                    
if DENSITY_BY_TRACER:
    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors', 'bytracers')
    output_dir = os.path.join(beta_dir, 'regressions', 'bytracers')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
else:
    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors')
    output_dir = os.path.join(beta_dir,'regressions')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

#by functional activity meassure 
for y_name in y_names:
    fname = f'{y_name}_{MASK_NAME}_regression_results_bysubject_{n_reg}.csv'
    results_df = pd.read_csv(os.path.join(output_dir, fname))

    #t-test
    t_test_results = []
    p_values = []
    for receptor in receptor_names:
        t_stat, p_value = ttest_1samp(results_df[receptor], 0)
        t_test_results.append({'receptor': receptor, 't_stat': t_stat, 'p_value': p_value})
        p_values.append(p_value)

    _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    significant_receptors = []
    for receptor, p_value_corrected in zip(receptor_names, p_values_corrected):
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
    ax.set_title(f'{MASK_NAME}: regression coefficients for {y_name} (FDR corrected)')

    # Add mean and standard deviation of R² to the plot
    textstr = f'Mean R²: {mean_R2:.2f}\nAdjusted mean R²: {mean_R2_adj:.2f}\nmean BIC: {mean_BIC:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.tight_layout()
    plt.show()
    
    fname = f'{y_name}_{MASK_NAME}_{n_reg}_reg_coef_groupedNeuroMod.png'
    fig_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir, fname), dpi=300)