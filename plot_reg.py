import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.legend_handler import HandlerTuple
from params_and_paths import Paths, Params, Receptors

PLOT_COEFS = True
PLOT_DOMINANCE = False 
ADD_CORR = False #if false, add the sign off the full regression instead
PLOT_LEGEND = False
ON_SURFACE = True

paths = Paths()
params = Params()
rec = Receptors()

if ON_SURFACE:
    proj = '_surf'
else:
    proj = ''


plt.rcParams.update({'font.size': 18})

if rec.source in ['PET', 'PET2']:
        receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.opioid, rec.glutamate, rec.histamine, rec.gaba, rec.dopamine, rec.cannabinnoid]
        group_names = ['serotonin', 'acetylcholine', 'norepinephrine', 'opioid', 'glutamate', 'histamine', 'gaba', 'cannabinnoid']
elif rec.source  == 'autorad_zilles44':
    receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.glutamate, rec.gaba, rec.dopamine]
elif rec.source == 'AHBA':
    receptor_groups = [rec.serotonin, rec.acetylcholine, rec.noradrenaline, rec.dopamine]
receptor_class = [rec.exc,rec.inh]

receptor_label_formatted = [
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
    '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$',
    '$5\\text{-}\\mathrm{HTT}$',
    '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$',
    '$\\mathrm{M}_{\\mathrm{1}}$',
    '$\\mathrm{VAChT}$',
    '$\\mathrm{NET}$',
    '$\\mathrm{A}_{\\mathrm{2}}$',
    '$\\mathrm{MOR}$',
    '$\\mathrm{m}\\mathrm{GluR}_{\\mathrm{5}}$',
    '$\\mathrm{NMDA}$',
    '$\\mathrm{H}_{\\mathrm{3}}$',
    '$\\mathrm{GABA}_{\\mathrm{a}}$',
    '$\\mathrm{D}_{\\mathrm{1}}$',
    '$\\mathrm{D}_{\\mathrm{2}}$',
    '$\\mathrm{DAT}$',
    '$\\mathrm{CB}_{\\mathrm{1}}$'
]

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
    mask_comb = params.mask + '_' + params.mask_details
else:
    mask_comb = params.mask 

output_dir = os.path.join(beta_dir, 'regressions', rec.source)
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if params.db == 'Explore':
    variables = ['surprise', 'confidence']
else:
    variables = params.latent_vars

if PLOT_COEFS:
    #plot regression coefficients
    for latent_var in variables:
        fname = f'{latent_var}_{mask_comb}_regression_results_bysubject_all{proj}_nearest.csv'
        results_df = pd.read_csv(os.path.join(output_dir, fname))
        if 'a2' in results_df.columns:
            results_df.rename(columns={'a2': 'A2'}, inplace=True)

        #t-test
        t_test_results = []
        p_values = []
        t_values = []
        for receptor in rec.receptor_names:
            t_stat, p_value = ttest_1samp(results_df[receptor], 0)
            t_test_results.append({'receptor': receptor, 't_stat': t_stat, 'p_value': p_value})
            p_values.append(p_value)
            t_values.append(t_stat)

        _, p_values_corrected = fdrcorrection(p_values, alpha=0.05)
        significant_receptors = []
        significant_receptors_signs = []
        for receptor, p_value_corrected, t in zip(rec.receptor_names, p_values_corrected, t_values):
            if p_value_corrected < 0.05:
                significant_receptors.append(receptor)
                significant_receptors_signs.append(np.sign(t))

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
                idx = significant_receptors.index(receptor)
                if significant_receptors_signs[idx] >0:
                    color = '#C96868'
                else:
                    color = '#7EACB5'
                ax.scatter(i, results_df[receptor].median(), color=color, zorder=5)  # Mark the median with a  dot

        ax.set_xticks(np.arange(len(ordered_receptors)))
        ax.set_xticklabels(receptor_label_formatted, rotation=90)
        for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
            group_idx = receptor_to_group.get(receptor, -1)
            label.set_color(base_colors[group_idx])
        ax.set_xlabel('Receptor')
        ax.set_ylabel('Coef Values')
        ax.set_title(f'{mask_comb}: regression coefficients for {latent_var} (FDR corrected) with {rec.source}', fontsize=12)

        # Add mean and standard deviation of R² to the plot
        textstr = f'Mean R²: {mean_R2:.2f}\nmean BIC: {mean_BIC:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        plt.tight_layout()
        plt.show()
        
        fname = f'{latent_var}_{mask_comb}_all_reg_coef{proj}_nearest.png'
        fig_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)


#plot dominance analysis 

if PLOT_DOMINANCE:
    for latent_var in variables:

        try:
            results_df = pd.read_pickle(os.path.join(output_dir, f'{latent_var}_{mask_comb}_dominance_allsubj.pickle'))
        except FileNotFoundError:
            print(f"File not found for {latent_var}, skipping...") #for now I only have the dominance data (computationally intensive) for surprise and confidence 
            continue


        if 'a2' in results_df.columns:
            results_df.rename(columns={'a2': 'A2'}, inplace=True)

        if ADD_CORR:
            fname = f'{latent_var}_{mask_comb}_correlation_results_bysubject.csv'
        else: 
            fname = f'{latent_var}_{mask_comb}_regression_results_bysubject_all.csv'
        
        sign_results = pd.read_csv(os.path.join(output_dir, fname))
        if 'a2' in sign_results.columns:
            sign_results.rename(columns={'a2': 'A2'}, inplace=True)
        sign_mean = sign_results.mean(axis=0) #mean correlation for each receptor

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

        plt.rcParams.update({'font.size': 18})

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
        ax.set_xticklabels(receptor_label_formatted, rotation=90)
        for label, receptor in zip(ax.get_xticklabels(), ordered_receptors):
            group_idx = receptor_to_group.get(receptor, -1)
            label.set_color(base_colors[group_idx])
        ax.set_xlabel('Receptor/Transporter')
        ax.set_ylabel('contribution (%)')

        if params.db == 'Explore':
            ax.set_ylim(0,0.145)
        #ax.set_title(f'{mask_comb}: dominance analysis for {latent_var} with {rec.source}', fontsize=12)

        # textstr = f'average R²: {mean_R2:.2f}'
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        #         verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        
        fname = f'{latent_var}_{mask_comb}_dominance.png'
        fig_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches='tight')

        #bigger writing for poster 
        for text in ax.get_figure().findobj(plt.Text):
            text.set_fontsize(28)

        fname = f'{latent_var}_{mask_comb}_dominance_poster.png'
        fig_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches='tight',transparent=True)

if PLOT_LEGEND:
    # Define the legend elements and labels
    legend_elements = []
    legend_labels = []

    # Add patches with adjacent light and dark squares for each receptor group
    for group_idx, (color, name) in enumerate(zip(base_colors, group_names)):
        dark_color = sns.dark_palette(color, n_colors=3)[1]
        light_color = sns.light_palette(color, n_colors=3)[1]
        
        # Create two small squares next to each other
        dark_square = mlines.Line2D([], [], color=dark_color, marker='s', markersize=10, linestyle='None')
        light_square = mlines.Line2D([], [], color=light_color, marker='s', markersize=10, linestyle='None')
        
        # Add the tuple of squares and the group name
        legend_elements.append((dark_square, light_square))
        legend_labels.append(name)

    # Add grey patches for excitatory/inhibitory distinction
    dark_grey_patch = mpatches.Patch(color="dimgray", label="excitatory")
    light_grey_patch = mpatches.Patch(color="lightgrey", label="inhibitory")
    legend_elements.extend([dark_grey_patch, light_grey_patch])
    legend_labels.extend(["excitatory", "inhibitory"])

    # Add a hatch pattern example for transporters
    hatch_example = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='transporter')
    legend_elements.append(hatch_example)
    legend_labels.append("transporter")

    # Add red and blue dots to indicate positive/negative correlations
    red_dot = mlines.Line2D([], [], color='#C96868', marker='o', markersize=8, linestyle='None', label="significant estimate in full model: positive")
    blue_dot = mlines.Line2D([], [], color='#7EACB5', marker='o', markersize=8, linestyle='None', label="significant estimate in full model: negative")
    legend_elements.extend([red_dot, blue_dot])
    legend_labels.extend(["sig. estimate in full model:\npositive", "sig. estimate in full model:\nnegative"])

    # Create a custom legend with both horizontal and vertical spacing
    fig, ax = plt.subplots(figsize=(2, 4))
    custom_legend = ax.legend(
        legend_elements, 
        legend_labels, 
        handler_map={
            tuple: HandlerTuple(ndivide=None, pad=0.8)  # Horizontal padding between squares
        },
        loc='center', ncol=1, frameon=False,
        labelspacing=1  # Vertical spacing between rows
    )
    ax.axis('off')

    #bigger writing for poster 
    for text in ax.get_figure().findobj(plt.Text):
        text.set_fontsize(20)

    fname = f'dominance_legend.png' 
    fig_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(os.path.join(fig_dir, fname),dpi=300, bbox_inches='tight',transparent=True)
    # Remove axis for clean legend display
    plt.show()



