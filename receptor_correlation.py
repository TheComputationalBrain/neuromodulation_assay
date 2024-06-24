import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, zscore, pearsonr
import main_funcs as mf
from params_and_paths import *


DENSITY_BY_TRACER = True

fmri_dir = mf.get_fmri_dir(DB_NAME)
subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

if DENSITY_BY_TRACER:
    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors', 'bytracers')
    output_dir = os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME,'first_level','regressions', 'bytracers')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
else:
    receptor_dir = os.path.join(home_dir[DATA_ACCESS], 'receptors')
    output_dir = os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME,'first_level','regressions')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
y_names = np.array(['surprise','confidence', 'predictability', 'predictions'])
X_data = zscore(np.load(os.path.join(home_dir[DATA_ACCESS],'receptors', f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True))

for y_name in y_names:
    results_df = pd.DataFrame(columns=receptor_names)

    for sub in subjects: 
        cors = np.zeros((len(receptor_names),))
        for indx,receptor in enumerate(receptor_names):

            y_data = np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten()
            non_nan_indices = ~np.isnan(y_data)
            X = X_data[:,indx]
            X = X[non_nan_indices]
            y = zscore(y_data[non_nan_indices])

            cors[indx] = pearsonr(X, y).correlation
            
        #results by functional activity across participants 
        results = pd.DataFrame([cors], columns = receptor_names)
        results_df = pd.concat([results_df,results], ignore_index=True)

    fname = f'{y_name}_{MASK_NAME}_correlation_results_bysubject.csv'
    results_df.to_csv(os.path.join(output_dir, fname), index=False)  

# plot correlation results in Boxplot
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

output_dir = os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME,'first_level','regressions')

#by fucntional activity meassure 
for y_name in y_names:
    fname = f'{y_name}_{MASK_NAME}_correlation_results_bysubject.csv'
    results_df = pd.read_csv(os.path.join(output_dir, fname))

    #t-test
    t_test_results = []
    significant_receptors = []
    for receptor in receptor_names:
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
    ax.set_title(f'{MASK_NAME}: correlation for {y_name}')
    plt.tight_layout()
    plt.show()
    
    fname = f'{y_name}_{MASK_NAME}_cor_coef_groupedNeuroMod.png'
    fig_dir = os.path.join(output_dir, 'plots_corr')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir, fname), dpi=300)