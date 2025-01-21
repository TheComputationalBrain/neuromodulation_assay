import os
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting, image
from nilearn import datasets
from nilearn import surface
from scipy.stats import norm, pearsonr
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import ttest_1samp
from params_and_paths import Paths, Params, Receptors

paths = Paths()
params = Params()
rec = Receptors()

vmax = -np.log10(1/100000) #1/n of permutations 
TRESH = -np.log10(0.05)
FWHM = 5
zscore_info = '' #"" #"_zscoreAll" ' or ""' #to explore how our zscoring decission influences the number and extend of sig clusters 

output_dir = os.path.join(paths.home_dir, 'domain_general')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

ANNOT_DIR = '/home_local/alice_hodapp/NeuroModAssay/domain_general/atlas'


###### plot the significant clusters by dataset and contrast indivisuallyindividually #####

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

#fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
fsaverage = datasets.fetch_surf_fsaverage()

for contrast in contrasts:
    if '_neg' in contrast:
        cmap = 'cool'
    else:
        cmap = 'hot'
    task_masks = []
    for task in tasks:
        group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
        img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass_{FWHM}.nii.gz')
        img = nib.load(img_path) 

        plotting.plot_img_on_surf(img, surf_mesh='fsaverage5', 
                                        hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=TRESH, vmax = vmax,
                                        title=contrast, colorbar=True,inflate=True, cmap= cmap, cbar_tick_format='%.2f')
        fname = f'{task}_{contrast}_thresh_cluster_mass{zscore_info}.png' 
        plt.savefig(os.path.join(output_dir, fname))

        texture = surface.vol_to_surf(img, fsaverage.pial_right)
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture, bg_map=fsaverage.sulc_right, hemi='right', threshold=TRESH, vmax = vmax, title=contrast, colorbar=True, cmap= cmap, cbar_tick_format='%.2f')
        
        fname = f'{task}_{contrast}_thresh_cluster_mass_right.png' 
        plt.savefig(os.path.join(output_dir, fname))


###### Plot overlap for all probability learning tasks #####

tasks = ['EncodeProb', 'NAConf', 'PNAS']
contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

with open(os.path.join(output_dir,'ROIs_overlap.txt'), "w") as outfile:

    for contrast in contrasts:
        rois_with_overlap = []
        for hemi in ['right', 'left']:
            task_masks = []
            for task in tasks:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass_{FWHM}.nii.gz')
                img = nib.load(img_path) 
                if hemi == 'right':
                    texture = surface.vol_to_surf(img, fsaverage.pial_right)  
                    mesh = fsaverage.infl_right
                    bg_map=fsaverage.sulc_right
                    annot_fname = 'rh.HCPMMP1.annot'
                else:
                    texture = surface.vol_to_surf(img, fsaverage.pial_left) 
                    mesh = fsaverage.infl_left
                    bg_map=fsaverage.sulc_left
                    annot_fname = 'lh.HCPMMP1.annot'
                task_masks.append(texture > TRESH)  

            task_masks = np.array(task_masks)
            encodeprob_mask = task_masks[0]
            naconf_mask = task_masks[1]
            pnas_mask = task_masks[2]
            all_tasks_mask = np.all(task_masks, axis=0)

            conditions_array = np.zeros(task_masks[0].shape, dtype=int)
            overlap_array = np.zeros(task_masks[0].shape, dtype=int)

            if 'confidence_neg' in contrast:
                conditions_array[pnas_mask] = 1  
                conditions_array[naconf_mask] = 2  
                conditions_array[encodeprob_mask] = 3  
                cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746'])
            else:
                conditions_array[naconf_mask] = 1  
                conditions_array[encodeprob_mask] = 2  
                conditions_array[pnas_mask] = 3    
                cmap = ListedColormap(['#FFC470', '#DD5746', '#4793AF'])          

            overlap_array[all_tasks_mask] = 1 #overlap map across all tasks

            annot_fpath = os.path.join(ANNOT_DIR, annot_fname)
            hcpmmp1_annot = nib.freesurfer.io.read_annot(annot_fpath)
            annot_reference_labels = hcpmmp1_annot[0]
            annot_reference_names = [name.decode('UTF-8') for name in hcpmmp1_annot[2]]
            indices_overlap = np.where(overlap_array==1)[0]
            levels = np.unique([annot_reference_labels[i] for i in indices_overlap])

            #full glasser parcelation
            figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, hemi=hemi, view='lateral', alpha=None)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=annot_reference_labels,
                hemi=hemi,
                # labels=None, #label the ROIs
                # levels=indices_overlap,
                figure=figure,
                cmap='Pastel2',
                legend=True)            
            fname = f'{contrast}_allOverlap_allProb_surface_{hemi}_HCPMMP1.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

            #ROI with overlap only
            figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, hemi=hemi, view='lateral', alpha=None)
            fname = f'{contrast}_allOverlap_allProb_surface_{hemi}.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=annot_reference_labels,
                hemi=hemi,
                levels=levels,
                figure=figure,
                cmap='Pastel2',
                legend=True)            
            fname = f'{contrast}_allOverlap_allProb_surface_{hemi}_HCPMMP1_ROIs.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

            rois_with_overlap.append(
                np.unique(
                    [annot_reference_names[annot_reference_labels[i]]
                        for i in indices_overlap]
                    ))
            
        with open(os.path.join(output_dir, f'{contrast}_rois'), 'wb') as f:
            pickle.dump(rois_with_overlap, f)
            
        outfile.write(f'{contrast}: ROIs with overlap:\n\n')
        for ROI in np.concatenate(rois_with_overlap):
            outfile.write(ROI + "\n")
        outfile.write('\n\n')


#### Plot overlap for all datasets #######
tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

with open(os.path.join(output_dir,'ROIs_overlap_explore.txt'), "w") as outfile:

    for contrast in contrasts:
        rois_with_overlap = []
        for hemi in ['right', 'left']:
            task_masks = []
            for task in tasks:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass_{FWHM}.nii.gz')
                img = nib.load(img_path) 
                if hemi == 'right':
                    texture = surface.vol_to_surf(img, fsaverage.pial_right)  
                    mesh = fsaverage.infl_right
                    bg_map=fsaverage.sulc_right
                    annot_fname = 'rh.HCPMMP1.annot'
                else:
                    texture = surface.vol_to_surf(img, fsaverage.pial_left) 
                    mesh = fsaverage.infl_left
                    bg_map=fsaverage.sulc_left
                    annot_fname = 'lh.HCPMMP1.annot'
                task_masks.append(texture > TRESH)  

            task_masks = np.array(task_masks)
            encodeprob_mask = task_masks[0]
            naconf_mask = task_masks[1]
            pnas_mask = task_masks[2]
            explore_mask = task_masks[3]
            all_tasks_mask = np.all(task_masks, axis=0)

            conditions_array = np.zeros(task_masks[0].shape, dtype=int)
            overlap_array = np.zeros(task_masks[0].shape, dtype=int)

            if 'confidence_neg' in contrast:
                conditions_array[pnas_mask] = 1  
                conditions_array[naconf_mask] = 2  
                conditions_array[encodeprob_mask] = 3  
                conditions_array[explore_mask] = 4  
                cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746', '#FAC67A'])
            else:
                conditions_array[naconf_mask] = 1  
                conditions_array[explore_mask] = 2     
                conditions_array[encodeprob_mask] = 3  
                conditions_array[pnas_mask] = 4
                cmap = ListedColormap(['#FFC470', '#FAC67A', '#DD5746', '#4793AF'])          

            overlap_array[all_tasks_mask] = 1 #overlap map across all tasks

            annot_fpath = os.path.join(ANNOT_DIR, annot_fname)
            hcpmmp1_annot = nib.freesurfer.io.read_annot(annot_fpath)
            annot_reference_labels = hcpmmp1_annot[0]
            annot_reference_names = [name.decode('UTF-8') for name in hcpmmp1_annot[2]]
            indices_overlap = np.where(overlap_array==1)[0]
            levels = np.unique([annot_reference_labels[i] for i in indices_overlap])

            #full glasser parcelation
            figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, hemi=hemi, view='lateral', alpha=None)
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=annot_reference_labels,
                hemi=hemi,
                figure=figure,
                cmap='Pastel2',
                legend=True)            
            fname = f'{contrast}_allOverlap_explore_surface_{hemi}_HCPMMP1.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

            #ROI with overlap only
            figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, hemi=hemi, view='lateral', alpha=None)
            fname = f'{contrast}_allOverlap_explore_surface_{hemi}.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plotting.plot_surf_contours(
                surf_mesh=mesh,
                roi_map=annot_reference_labels,
                hemi=hemi,
                levels=levels,
                figure=figure,
                cmap='Pastel2',
                legend=True)            
            fname = f'{contrast}_allOverlap_explore_surface_{hemi}_HCPMMP1_ROIs.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

            rois_with_overlap.append(
                np.unique(
                    [annot_reference_names[annot_reference_labels[i]]
                        for i in indices_overlap]
                    ))
            
        with open(os.path.join(output_dir, f'{contrast}_rois'), 'wb') as f:
            pickle.dump(rois_with_overlap, f)
            
        outfile.write(f'{contrast}: ROIs with overlap:\n\n')
        for ROI in np.concatenate(rois_with_overlap):
            outfile.write(ROI + "\n")
        outfile.write('\n\n')


##### Plot colorbars for overlap analysis ######
categories = [0, 1, 2, 3]
labels = ["MarkovGuess", "NAConf", "EncodeProb", "Explore"]

cmap = cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746', '#FAC67A'])

# Set up normalization
norm = BoundaryNorm(categories + [len(categories)], cmap.N)  # Define boundaries for the colormap

# Create a dummy ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create the figure for the legend
fig, ax = plt.subplots(figsize=(0.5, 5))  # Adjust size as needed

# Add the colorbar
cbar = plt.colorbar(sm, cax=ax, orientation='vertical')

# Adjust the ticks to be centered in the middle of the colors
tick_positions = [0.5, 1.5, 2.5, 3.5]  # Positions centered within each color
cbar.set_ticks(tick_positions)
cbar.ax.set_yticklabels(labels)  # Set custom labels

# Style the colorbar
cbar.ax.tick_params(labelsize=10)  # Adjust label font size
fname = 'colorbar_overlap_4cat-png'
plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)


##### Plot group level correleation heatmap ######


tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
cmap = np.genfromtxt('/home/ah278717/hansen_receptors/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

# Initialize a list to store all data
data_list = []
for task in tasks:  # Loop through task folders
    task_path = os.path.join(paths.home_dir, task, params.mask,'second_level')
    for contrast_file in os.listdir(task_path):  # Loop through subject files
        contrast_path = os.path.join(task_path, contrast_file)
        if contrast_file.endswith(f"group_confidence_schaefer_effect_size{zscore_info}.pickle"):  
            confidence_array = np.load(contrast_path, allow_pickle=True).flatten()
            data_list.append({'task': task, 'contrast': 'confidence', 'data': confidence_array})
        elif contrast_file.endswith(f"group_surprise_schaefer_effect_size{zscore_info}.pickle"):  # Load surprise arrays
            surprise_array = np.load(contrast_path, allow_pickle=True).flatten()
            data_list.append({'task': task, 'contrast': 'surprise', 'data': surprise_array})

all_data = pd.DataFrame(data_list)

# Initialize a correlation results matrix
n = len(all_data)
correlation_matrix = np.zeros((n, n))

# Calculate pairwise correlations
for i in range(n):
    for j in range(n):
        data_i = all_data.iloc[i]['data']
        data_j = all_data.iloc[j]['data']
        correlation_matrix[i, j] = pearsonr(data_i, data_j)[0]  

# Create a mapping of indices to (task, subject, contrast)
labels = [f"{row['task']}_{row['contrast']}" for _, row in all_data.iterrows()]

correlation_df = pd.DataFrame(correlation_matrix, index=labels, columns=labels)

explore_indices = [i for i, label in enumerate(labels) if "Explore_confidence" in label]
for i in explore_indices:
    correlation_matrix[i, :] *= -1
    correlation_matrix[:, i] *= -1

explore_labels = [label for label in labels if "Explore" in label]
non_explore_labels = [label for label in labels if "Explore" not in label]
reordered_labels = non_explore_labels + explore_labels
reordered_indices = [labels.index(label) for label in reordered_labels]
correlation_matrix_reordered = correlation_matrix[np.ix_(reordered_indices, reordered_indices)]

correlation_df_reordered = pd.DataFrame(
    correlation_matrix_reordered, index=reordered_labels, columns=reordered_labels
)

correlation_df_reordered.to_csv(os.path.join(output_dir, 'correlation_df.csv'))
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df_reordered, annot=True, fmt=".2f", cmap=cmap_div, square=True, center=0)
fname = 'group_correlation_heatmap.png'
plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)

def custom_sort(row):
    if row['contrast'] == 'confidence':
        return (0, row['task'] == 'Explore', row['task'])  # Confidence first, Explore last within confidence
    else:  # For 'surprise'
        return (1, row['task'] == 'Explore', row['task'])  # Surprise next, Explore last within surprise

all_data_sorted = all_data.sort_values(by=['contrast', 'task'], key=lambda col: all_data.apply(custom_sort, axis=1))

sorted_labels = [f"{row['task']}_{row['contrast']}" for _, row in all_data_sorted.iterrows()]
reordered_indices = [labels.index(label) for label in sorted_labels]
correlation_matrix_reordered = correlation_matrix[np.ix_(reordered_indices, reordered_indices)]
explore_indices = [i for i, label in enumerate(sorted_labels) if "Explore_confidence" in label]
correlation_df_reordered = pd.DataFrame(
    correlation_matrix_reordered, index=sorted_labels, columns=sorted_labels
)
correlation_df_reordered.to_csv(os.path.join(output_dir, 'correlation_df_reordered.csv'))
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df_reordered, annot=True, fmt=".2f", cmap=cmap_div, square=True, center=0)
fname = 'group_correlation_heatmap_reordered.png'
plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)


