# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:02:11 2022

@author: Saeyeon, Alice

This script is self contained, as it load already existing meta analysis data and is supposed to also run independendly of this project
"""

# Import useful packages 

'''import packages from python'''
import pickle
import re
import types
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import seaborn as sns
from pathlib import Path
from scipy.stats import  ttest_1samp
from sklearn.linear_model import LinearRegression

# --- Patch deprecated NumPy aliases ---
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object

# --- Patch missing pandas index types for old pickles ---
fake_numeric = types.ModuleType("pandas.core.indexes.numeric")
fake_numeric.Int64Index = type(pd.Index([1, 2, 3]))
fake_numeric.RangeIndex = type(pd.RangeIndex(3))
sys.modules['pandas.core.indexes.numeric'] = fake_numeric


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from TransitionProbModel.MarkovModel_Python import IdealObserver as IO
from TransitionProbModel.MarkovModel_Python import GenerateSequence as sg


#Compute some useful functions :
    
def entropy(p):
    p = np.clip(p, np.finfo(np.float).resolution, 1-np.finfo(np.float).resolution) 
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


def function_indice_change_point(list, val):
    if len(list) < 1:#if the list is empty
        l = [val]
    elif min(list) >= val:
        l = [val]
    else :
        l = [list[i] for i in range (len(list)) if list[i] <= val]
    return(max(l))

def print_stat_results(df, col, file, label=''):
    collumn = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
    t_res = ttest_1samp(collumn, 0)
    file.write(f"{label}{col} Correlation:\n")
    file.write(f"{label}{col}: rho={df[col].mean():.02f}, " +
                f"SD={df[col].std():.03f}, " +
                f"SEM={df[col].sem():.03f}, " +
                f"d={df[col].mean() / df[col].std():.03f}, " +
                f"t({df[col].count()-1})={t_res.statistic:.01f}, " +
                f"p={t_res.pvalue:.1e}")
    file.write('\n\n')

def print_regression_results(df, col, file, label=''):
    # R²
    r2_values = pd.to_numeric(df[f"{col}_r2"], errors='coerce').dropna().to_numpy(dtype=float)
    r2_mean = r2_values.mean()
    r2_std = r2_values.std()
    
    # Slope (beta coefficient)
    slopes = pd.to_numeric(df[f"{col}_slope"], errors='coerce').dropna()
    t_res = ttest_1samp(slopes, 0)
    
    file.write(f"{label}{col} Regression:\n")
    file.write(f"  R²: mean={r2_mean:.03f}, SD={r2_std:.03f}\n")
    file.write(f"  Slope: mean={slopes.mean():.03f}, SD={slopes.std():.03f}, "
               f"d={slopes.mean()/slopes.std():.03f}, "
               f"t({slopes.count()-1})={t_res.statistic:.02f}, p={t_res.pvalue:.2e}\n\n")
    
def set_publication_style(
    font_size=7,
    line_width=1,
    context="paper",
    layout="single",
    page="full",
):
    """
    Set consistent, publication-quality figure style 
    
    Parameters
    ----------
    font_size : int
        Base font size for all text.
    line_width : float
        Default line width for axes and lines.
    context : str
        Seaborn context ('paper', 'notebook', 'talk', 'poster').
    layout : str
        'single', '2-across', '3-across', or '6-across'
    page : str
        'single' (column width ≈ 8.9 cm) or 'full' (page width ≈ 17.8 cm)
    """

    import matplotlib
    import seaborn as sns

    cm_to_inch = 1 / 2.54

    # --- PNAS-style page widths ---
    page_widths = {
        "single": 8.9 * cm_to_inch,   # ≈3.5"
        "full": 17.8 * cm_to_inch,    # ≈7.0"
    }

    if page not in page_widths:
        raise ValueError("page must be 'single' or 'full'")

    page_width = page_widths[page]
    gutter = 0.25 * cm_to_inch  # ≈0.1" between panels

    # --- Compute figure width ---
    n_panels = {
        "single": 1,
        "2-across": 2,
        "3-across": 3,
        "6-across": 6,
    }.get(layout)

    if n_panels is None:
        raise ValueError("layout must be 'single', '2-across', '3-across', or '6-across'")

    total_gutter = (n_panels - 1) * gutter
    fig_width = (page_width - total_gutter) / n_panels

    # Aspect ratio — for 6-across, use a shallower height
    if layout == "6-across":
        fig_height = fig_width * 0.65
    else:
        fig_height = fig_width * 0.75

    # Error bar cap scaling
    capsize = {
        "single": 2.5,
        "2-across": 2.0,
        "3-across": 1.5,
        "6-across": 1.0,
    }[layout]
    matplotlib.rcParams["errorbar.capsize"] = capsize

    # --- Apply styling ---
    sns.set_theme(
        style="ticks",
        context=context,
        font="sans-serif",
        rc={
            "figure.figsize": (fig_width, fig_height),
            "figure.dpi": 300,

            # Fonts
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,

            # Lines & ticks
            "axes.linewidth": 0.5,
            "lines.linewidth": line_width,
            "lines.markersize": 3,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,

            # Export-friendly
            "savefig.transparent": True,
        },
    )

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'



#  Get data and inference for each subject
behavior_path = '/neurospin/unicog/protocols/comportement/ConfidenceDataBase_2020_Meyniel/subjects_data'
out_dir = '/home_local/alice_hodapp/NeuroModAssay/figures'

#subject to skip 
naconf_numbers = [3, 5, 6, 9, 36, 51]
encodeprob_numbers = [1, 4, 12, 20]

#create a dataframe 
data = pd.DataFrame(
    columns=['subject', 'sub_pred', 'sub_conf', 'io_pred', 'io_conf',
             'io_surp', 'io_entropy', 'res_sub_conf', 'samples_conf'])

#open all subjects information dataframe

filename_info = '/neurospin/unicog/protocols/comportement/ConfidenceDataBase_2020_Meyniel/all_subjects_info.pickle'
file_info = open(filename_info, 'rb')
all_subjects_info = pickle.load(file_info, encoding='bytes')

#list of the names of all the subjects 
list_subject = [f for f in os.listdir(behavior_path) if f.endswith(".pickle")]


valid_tasks = ["EncodeProb2020", "NACONFfMRI", "PNAS2017"]

# Define the datasets to keep
filtered_list_subject = [
    subject for subject in list_subject
    if any(valid_task in subject for valid_task in valid_tasks)  # Keep subjects that match valid values
    and not (
        # Extract valid task (first part before first _) and number (part after last _ and before .pickle)
        (
            # For "NACONFfMRI" task, extract number and check if it matches with naconf_numbers
            ("NACONFfMRI" in subject and "NACONFfMRI" in subject.split('_')[0] and
            int(subject.split('_')[-1].replace('.pickle', '')) in naconf_numbers) or
            
            # For "EncodeProb" task, extract number and check if it matches with encodeprob_numbers
            ("EncodeProb" in subject and "EncodeProb" in subject.split('_')[0] and
            int(subject.split('_')[-1].replace('.pickle', '')) in encodeprob_numbers)
        )
    )  # Skip subjects based on the number condition
]

for subject in filtered_list_subject:    
    # Extract the number after the last underscore
    match = re.search(r'_(\d{3})\.pickle$', subject)
    if match:
        number = int(match.group(1))  # Convert the captured number to an integer
    
    with open(os.path.join(behavior_path, subject), 'rb') as f:
        sub = pickle.load(f, encoding='bytes')
    
    #IO estimates
    io_pred, io_conf, io_surp, io_entropy = [], [], [], []
    question_indices = {}
    out_hmm = {}
    
    number_of_samples = []
    generative_p = {}
    indices_chunk = {}
    
    #define the structure (order) of the sequence for the given subject
    if 'markov' in all_subjects_info.loc[all_subjects_info['sub_id'] == subject.replace('.pickle', '')]['structure'].values[0]:
        order_prob = 1
    elif 'bernoulli' in all_subjects_info.loc[all_subjects_info['sub_id'] == subject.replace('.pickle', '')]['structure'].values[0]:
        order_prob = 0
    
    #iterate over the sessions 
    for (k,session) in enumerate(sub):
        question_indices[k] = session['question_indices']
        question_indices[k] = question_indices[k].astype(int)
        if question_indices[k][-1] >= len(session['sequence']): 
            question_indices[k] = question_indices[k][:-1]
        
        #compute IO 
        options = {'p_c': session['volatility'], 'resol': 20}
        out_hmm[k] = IO.IdealObserver(session['sequence'], 'hmm', order=order_prob, options=options)
        
        #compute IO predictions, confidences, surprises 
        for q_index in question_indices[k]:
            if order_prob == 0: 
                io_pred.append(out_hmm[k][(1,)]['mean'][q_index])
                io_conf.append(- np.log(out_hmm[k][(1,)]['SD'][q_index]))
                io_surp.append(out_hmm[k]['surprise'][q_index]) 
            if order_prob == 1:         
                if session['sequence'][q_index] == 0:
                    io_pred.append(out_hmm[k][(0,1)]['mean'][q_index])
                    io_conf.append(- np.log(out_hmm[k][(0,1)]['SD'][q_index]))
                    io_surp.append(out_hmm[k]['surprise'][q_index])
                else :
                    io_pred.append(out_hmm[k][(1,1)]['mean'][q_index])
                    io_conf.append(- np.log(out_hmm[k][(1,1)]['SD'][q_index]))
                    io_surp.append(out_hmm[k]['surprise'][q_index])
            
        #compute number of samples since the last change point
        generative_p[k] = session['generative p(1)']# list of generative probabilities for a session 
        indices = [] 
        for p in range (len(generative_p[k])-1):
            if generative_p[k][p] != generative_p[k][p+1]:
                indices.append(p)
        indices_chunk[k]=indices
                
        #count number of observations since the last change point      
        for element in question_indices[k]:
            indice_change_point = function_indice_change_point((indices_chunk[k]), element)
            #indices_chunk is a dict with keys = number of sessions and values for each key = chunks' indices for each session 
            #indice_change_point is the indice of the last change point before before the question indice OR the question indice itself  
            if element != indice_change_point:
                nb_observations = element - indice_change_point 
            else : 
                nb_observations = element + 1
            number_of_samples.append(np.log(nb_observations))
        
        #convert indices into float for practical purposes              
        samples = [float(i) for i in number_of_samples]
    
    
    #compute entropy based on IO probability estimates 
    for proba_io in io_pred :
        io_entropy.append(entropy(proba_io))  

    n_questions = len(io_pred)               

    if 'PNAS' in subject:
        sub_pred = nan_array = np.nan * np.ones(n_questions) #MarkovGuess did not have a a probbaility report 
    else:
        sub_pred = np.hstack([session['prediction_1'].tolist() for session in sub])  #prediction_1 is the next item being 1
        if 'NACONF' in subject:
            sub_pred = 1 - sub_pred

    #create a dataframe for the subject            
    sub_data = pd.DataFrame({
    'subject': [subject]*n_questions,
    'sub_pred': sub_pred,  
    'sub_conf': np.hstack([session['confidence'].tolist() for session in sub]),
    'io_pred': np.array(io_pred),
    'io_conf': np.array(io_conf),
    'io_surp': np.array(io_surp),
    'io_entropy': np.array(io_entropy),
    'res_sub_conf': [np.nan]*n_questions,
    'samples_conf': np.array(samples)
    })
        
    #compute residuals     
    #drop rows with NaN in relavant collumns
    sub_data = sub_data.dropna(subset=['io_surp', 'io_entropy', 'sub_conf'])
    sub_data.reset_index(inplace = True)
    
    X = np.vstack([sub_data['io_surp'].values,
                               sub_data['io_entropy'].values,
                               sub_data['samples_conf'].values]).T
    reg = LinearRegression().fit(X, sub_data['sub_conf'])
    sub_data['res_sub_conf'] = sub_data['sub_conf'] - reg.predict(X) 
    
    # concatenate dataframes 
    data = pd.concat([data, sub_data])    

    print(io_conf)


#  Compute within subject correlations and linear regressions

subj_correlations = pd.DataFrame(index=filtered_list_subject,
                                 columns=['sub_io_conf', 'sub_io_conf_int', 'sub_io_conf_slope',
                                          'sub_io_pred', 'sub_io_pred_int', 'sub_io_pred_slope',
                                          'res_sub_io_conf', 'res_sub_io_conf_int', 'res_sub_io_conf_slope', 
                                          'sub_samples_conf', 'sub_samples_conf_int', 'sub_samples_conf_slope'
                                          ])
for subject in filtered_list_subject:
    print(subject)
    sub_data = data[data['subject'] == subject]
    for (x, y, var) in zip(['io', 'io', 'io', 'samples'],
                           ['sub', 'res_sub', 'sub', 'sub'],
                           ['conf', 'conf', 'pred', 'conf']):
        
        if (var == 'pred' and 'PNAS' in subject):
            subj_correlations.loc[subject, "_".join([y, x, var])] = np.nan
            subj_correlations.loc[subject, "_".join([y, x, var, 'int'])] = np.nan
            subj_correlations.loc[subject, "_".join([y, x, var, 'slope'])] = np.nan
        else:
            clean_data = sub_data.dropna(subset=["_".join([y, var])]) 
            # correlation
            subj_correlations.loc[subject, "_".join([y, x, var])] = \
                np.corrcoef(clean_data["_".join([x, var])], clean_data["_".join([y, var])])[0, 1]

            X = clean_data["_".join([x, var])].to_numpy()[:, np.newaxis]
            Y = clean_data["_".join([y, var])].to_numpy()
            reg = LinearRegression().fit(X, Y)

            # store intercept and slope
            subj_correlations.loc[subject, "_".join([y, x, var, 'int'])] = reg.intercept_
            subj_correlations.loc[subject, "_".join([y, x, var, 'slope'])] = reg.coef_[0]

            # store R²
            r2 = reg.score(X, Y)
            subj_correlations.loc[subject, "_".join([y, x, var, 'r2'])] = r2


#### group level output ######
''' t-tests'''
#print to text file 
with open(os.path.join(out_dir, 'behav_group_summary_meta.txt'), "w") as file:

    for col in ['sub_io_pred', 'sub_io_conf']:
        print_stat_results(subj_correlations, col, file)
        print_regression_results(subj_correlations, col, file)
        
#  Plot group level effects, binned, with linear fit
set_publication_style(font_size=7, layout="6-across")

N_BINS = 6
analysis_list = [
    {'dep_var': 'sub_pred', 'ind_var': 'io_pred', 'reg': 'sub_io_pred',
     'x_label': 'Ideal probability \nestimate', 'y_label': 'Subj. prob. est.'},
    {'dep_var': 'sub_conf', 'ind_var': 'io_conf', 'reg': 'sub_io_conf',
     'x_label': 'Ideal confidence \n(log precision)', 'y_label': 'Subj. conf.'}
    ]

for analysis in analysis_list:
    binned_data = data.groupby(by=[pd.qcut(data[analysis['ind_var']].rank(method='first'), N_BINS), 'subject']).mean()
    binned_data = binned_data.groupby(level=analysis['ind_var'])
    fig, ax = plt.subplots()
    #simple_axis(ax)
    if analysis['dep_var'] == 'io_pred':
        xlim = [0, 1]
    else:
        xlim = np.array([binned_data.mean()[analysis['ind_var']].min() - 0.1,
                         binned_data.mean()[analysis['ind_var']].max() + 0.1])
    if analysis['dep_var'] == 'sub_conf':
        ls = '--'
    else:
        ls = '-'
    #plot of the reg line
    ax.plot(xlim,
             xlim * subj_correlations[analysis['reg'] + '_slope'].mean() +
             subj_correlations[analysis['reg'] + '_int'].mean(),
             'k',
             lw=3,
             color="darkgrey",
             zorder=1,
             ls=ls)
    ax.errorbar(binned_data.mean()[analysis['ind_var']],
                 binned_data.mean()[analysis['dep_var']],
                 binned_data.sem()[analysis['dep_var']],
                 fmt='o', 
                 color="black",
                 zorder=2)
    ax.set_xlabel(analysis['x_label'])
    ax.set_ylabel(analysis['y_label'])
    ax.tick_params(axis='both', which='major', pad=8)
    if analysis['x_label'] == 'Ideal confidence \n(log precision)' and analysis['y_label'] == 'Subj. conf.':
        ax.set_ylim(0.5, 0.8)
    sns.despine(trim=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{analysis['dep_var']}_vs_{analysis['ind_var']}_group.svg"))
    fig.savefig(os.path.join(out_dir, f"{analysis['dep_var']}_vs_{analysis['ind_var']}_group.png"), dpi=300)


###### by study output ######

errorbar_kwargs = dict(
    fmt='o',
    color='black',
    zorder=2,
    elinewidth=0.5,   # thicker error lines
    capsize=2.0,      # small horizontal caps for visibility
    markersize=2,   # slightly smaller than default marker size
)

data['dataset'] = data['subject'].str.split('_').str[0]

# Create a new column or group identifier from the index
subj_correlations['dataset'] = subj_correlations.index.to_series().str.split('_').str[0]

# Group by subject
grouped = subj_correlations.groupby('dataset')

with open(os.path.join(out_dir, 'behav_group_summary_bystudy.txt'), "w") as file:
    for dataset, group_df in grouped:
        file.write(f"==== dataset: {dataset} ====\n")
        for col in ['sub_io_pred', 'sub_io_conf']:
            if (dataset == 'PNAS2017' and col == 'sub_io_pred'):
                continue
            # Only process if column exists and is numeric
            if col in group_df.columns:
                print_stat_results(group_df, col, file)
                print_regression_results(group_df, col, file)

#plotting
# Get dataset names from 'subject' column (everything before the first '_')
data['dataset'] = data['subject'].str.extract(r'(^[^_]+)')
subj_correlations.index.name = 'subject'
subj_correlations = subj_correlations.reset_index()  # Add 'subject' column globally
subj_correlations['dataset'] = subj_correlations['subject'].str.split('_').str[0]

for dataset_name, dataset_df in data.groupby('dataset'):
    for analysis in analysis_list:
        # Skip condition for specific dataset and dep_var
        if dataset_name == 'PNAS2017' and analysis['dep_var'] == 'io_pred':
            continue
        
        # Bin per subject
        binned = dataset_df.groupby(
            [pd.qcut(dataset_df[analysis['ind_var']].rank(method='first'), N_BINS), 'subject']
        ).mean(numeric_only=True)  # Ensure we only aggregate numeric columns
        
        # Then average across subjects within each bin
        binned_mean = binned.groupby(level=0).mean()
        binned_sem = binned.groupby(level=0).sem()

        fig, ax = plt.subplots()
        if analysis['dep_var'] == 'io_pred':
            xlim = [0, 1]
        else:
            x_vals = binned_mean[analysis['ind_var']]
            xlim = np.array([x_vals.min() - 0.1, x_vals.max() + 0.1])

        ls = '--' if analysis['dep_var'] == 'sub_conf' else '-'

        # Regression line: average slope/intercept across subjects in this dataset
        subset_corrs = subj_correlations.loc[
            subj_correlations['subject'].str.startswith(dataset_name)
        ]
        slope_mean = subset_corrs[analysis['reg'] + '_slope'].mean()
        int_mean = subset_corrs[analysis['reg'] + '_int'].mean()

        ax.plot(xlim, slope_mean * np.array(xlim) + int_mean,
                lw=1, color="darkgrey", zorder=1, ls=ls)

        ax.errorbar(
            binned_mean[analysis['ind_var']],
            binned_mean[analysis['dep_var']],
            binned_sem[analysis['dep_var']],
            **errorbar_kwargs
        )

        ax.set_xlabel(analysis['x_label'])
        ax.set_ylabel(analysis['y_label'])
        ax.tick_params(axis='both', which='major', pad=3)

        if analysis['x_label'] == 'Ideal confidence \n(log precision)' and analysis['y_label'] == 'Subj. conf.':
            # Calculate the min and max of the error bars
            lower_bound = binned_mean[analysis['dep_var']] - binned_sem[analysis['dep_var']]
            upper_bound = binned_mean[analysis['dep_var']] + binned_sem[analysis['dep_var']]
            
            # Set ymin and ymax with 0.1 padding below the minimum and above the maximum
            ymin = lower_bound.min() - 0.05
            ymax = upper_bound.max() + 0.05
            
            # Apply the new limits
            ax.set_ylim(ymin, ymax)

        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        sns.despine(trim=False)


        filename = f"{dataset_name}_{analysis['dep_var']}_vs_{analysis['ind_var']}_group.svg"
        fig.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
        plt.close(fig)