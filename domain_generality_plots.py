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
from nilearn.surface import SurfaceImage
from nilearn.glm import threshold_stats_img
from scipy.stats import norm, pearsonr
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import ttest_1samp
from params_and_paths import Paths, Params, Receptors
from nilearn.datasets import fetch_atlas_surf_destrieux


paths = Paths()
params = Params()

RESOLUTION = 'fsaverage' #change freesurfer5 to freesurfer to get high res surface plots

# what to plot
IDV_CLUSTER = False
IDV_FDR = False
CLUSTER_OVERLAP_PROB = False
CLUSTER_OVERLAP_ALL = False
CBAR_CLUSTER_OVERLAP_PROB = False
CBAR_CLUSTER_OVERLAP_ALL = False
CORR_PROB = False
CORR_ALL = True

# settings for plots
EXPLORE_MODEL = 'noEntropy_noER'
VMAX = -np.log10(1/100000) #1/n of permutations 
TRESH = -np.log10(0.05)
FWHM = 5
zscore_info = '' #"" #"_zscoreAll" ' or ""' #to explore how our zscoring decission influences the number and extend of sig clusters 

output_dir = os.path.join(paths.home_dir, 'domain_general')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

ANNOT_DIR = '/home_local/alice_hodapp/NeuroModAssay/domain_general/atlas'

fsaverage = datasets.fetch_surf_fsaverage(mesh=RESOLUTION)

###### plot the significant clusters by dataset and contrast indivisually #####

if IDV_CLUSTER:
    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
    contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

    for contrast in contrasts:
        if '_neg' in contrast:
            cmap = 'Blues'
        else:
            cmap = 'Reds'
        task_masks = []
        for task in tasks:
            if task == 'Explore':
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
            else:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
            
            if task in ['NAConf']:
                if params.remove_trials:
                    add_info = '_firstTrialsRemoved'
            elif not params.zscore_per_session:
                add_info = '_zscoreAll'
            else:
                add_info = ""

            img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
            img = nib.load(img_path) 

            plotting.plot_img_on_surf(img, surf_mesh=RESOLUTION, 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=TRESH, vmax = VMAX,
                                            title=contrast, colorbar=True,inflate=True, cmap= cmap, cbar_tick_format='%.2f')
            fname = f'{task}_{contrast}_thresh_cluster_mass.png' 
            plt.savefig(os.path.join(output_dir, 'plots_by_task', fname))

            texture = surface.vol_to_surf(img, fsaverage.pial_right)
            plotting.plot_surf_stat_map(
                fsaverage.infl_right, texture, bg_map=fsaverage.sulc_right, hemi='right', threshold=TRESH, vmax = VMAX, title=contrast, colorbar=True, cmap= cmap, cbar_tick_format='%.2f')
            
            fname = f'{task}_{contrast}_thresh_cluster_mass_right.png' 
            plt.savefig(os.path.join(output_dir, 'plots_by_task', fname))

            plt.close('all')

##### plot FDR corrected maps #####
if IDV_FDR:
    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
    contrasts = ['surprise', 'confidence']

    for contrast in contrasts:
    # Initialize arrays to store masks
        task_masks = []
        # Loop through tasks and load data
        for task in tasks:
            if task == 'Explore':
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
            else:
                group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
            img_path = os.path.join(group_dir, f'{contrast}_schaefer_z_map{zscore_info}.nii.gz')
            img = nib.load(img_path) 

            _, threshold = threshold_stats_img(img, alpha=0.05, height_control="fdr")

            plotting.plot_img_on_surf(img, surf_mesh=RESOLUTION, 
                                            hemispheres=['left', 'right'], views=['lateral', 'medial'], threshold=threshold,
                                            title=contrast, colorbar=True,inflate=True, cmap= 'coolwarm', cbar_tick_format='%.2f')
            fname = f'{task}_{contrast}_thresh_FDR.png' 
            plt.savefig(os.path.join(output_dir, 'plots_by_task', fname))

            texture = surface.vol_to_surf(img, fsaverage.pial_right)
            plotting.plot_surf_stat_map(
                fsaverage.infl_right, texture, bg_map=fsaverage.sulc_right, hemi='right', threshold=threshold, title=contrast, colorbar=True, cmap= 'coolwarm', cbar_tick_format='%.2f')
            
            fname = f'{task}_{contrast}_thresh_FDR_right.png' 
            plt.savefig(os.path.join(output_dir, 'plots_by_task', fname))

            plt.close('all')


###### Plot overlap for all probability learning tasks #####

if CLUSTER_OVERLAP_PROB:
    tasks = ['EncodeProb', 'NAConf', 'PNAS']
    contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

    with open(os.path.join(output_dir,'ROIs_overlap.txt'), "w") as outfile:

        for contrast in contrasts:
            rois_with_overlap = []
            for hemi in ['right', 'left']:
                task_masks = []
                for task in tasks:
                    if task == 'Explore':
                        group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
                    else:
                        group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')

                    if task in ['NAConf']:
                        if params.remove_trials:
                            add_info = '_firstTrialsRemoved'
                    elif not params.zscore_per_session:
                        add_info = '_zscoreAll'
                    else:
                        add_info = ""
            
                    img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
                    img = nib.load(img_path) 
                    if hemi == 'right':
                        texture = surface.vol_to_surf(img, fsaverage.pial_right)  
                        mesh = fsaverage.infl_right
                        bg_map=fsaverage.sulc_right
                        annot_fname = 'rh.HCP-MMP1.annot'
                    else:
                        texture = surface.vol_to_surf(img, fsaverage.pial_left) 
                        mesh = fsaverage.infl_left
                        bg_map=fsaverage.sulc_left
                        annot_fname = 'lh.HCP-MMP1.annot'
                    task_masks.append(texture > TRESH)  

                # task_masks = np.array(task_masks)
                # encodeprob_mask = task_masks[0]
                # naconf_mask = task_masks[1]
                # pnas_mask = task_masks[2]
                all_tasks_mask = np.all(task_masks, axis=0)

                conditions_array = np.zeros(task_masks[0].shape, dtype=int)
                overlap_array = np.zeros(task_masks[0].shape, dtype=int)

                cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746', '#FAC67A'])

                # if 'confidence_neg' in contrast:
                #     conditions_array[pnas_mask] = 1  
                #     conditions_array[encodeprob_mask] = 2  
                #     conditions_array[naconf_mask] = 3  
                #     cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746'])
                # else:
                #     conditions_array[naconf_mask] = 1  
                #     conditions_array[encodeprob_mask] = 2  
                #     conditions_array[pnas_mask] = 3    
                #     cmap = ListedColormap(['#DD5746', '#FFC470', '#4793AF']) 

                conditions_array = np.sum(task_masks, axis=0)
                overlap_array[all_tasks_mask] = 1 #overlap map across all tasks

                annot_fpath = os.path.join(ANNOT_DIR, annot_fname)
                hcpmmp1_annot = nib.freesurfer.io.read_annot(annot_fpath)
                annot_reference_labels = hcpmmp1_annot[0]
                annot_reference_names = [name.decode('UTF-8') for name in hcpmmp1_annot[2]]
                indices_overlap = np.where(overlap_array==1)[0]
                levels = np.unique([annot_reference_labels[i] for i in indices_overlap])

                #full glasser parcelation
                figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, vmin=1, vmax=3, hemi=hemi, view='lateral', alpha=None)
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
                figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, vmin=1, vmax=3,cmap=cmap, hemi=hemi, view='lateral', alpha=None)
                fname = f'{contrast}_allOverlap_allProb_surface_{hemi}.png'
                plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

                plotting.plot_surf_contours(
                    surf_mesh=mesh,
                    roi_map=annot_reference_labels,
                    hemi=hemi,
                    levels=levels,
                    labels = annot_reference_names,
                    figure=figure,
                    cmap='Set2',
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

            plt.close('all')

if CLUSTER_OVERLAP_ALL:
    #### Plot overlap for all datasets #######
    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
    contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

    with open(os.path.join(output_dir,'ROIs_overlap_explore.txt'), "w") as outfile:

        for contrast in contrasts:
            rois_with_overlap = []
            for hemi in ['right', 'left']:
                task_masks = []
                for task in tasks:
                    if task in ['NAConf']:
                        add_info = '_firstTrialsRemoved'
                    else:
                        add_info = ""
        
                    if task == 'Explore':
                        group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
                    else:
                        group_dir = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                    img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
                    img = nib.load(img_path) 
                    if hemi == 'right':
                        texture = surface.vol_to_surf(img, fsaverage.pial_right)  
                        mesh = fsaverage.infl_right
                        bg_map=fsaverage.sulc_right
                        annot_fname = 'rh.HCP-MMP1.annot'
                    else:
                        texture = surface.vol_to_surf(img, fsaverage.pial_left) 
                        mesh = fsaverage.infl_left
                        bg_map=fsaverage.sulc_left
                        annot_fname = 'lh.HCP-MMP1.annot'
                    task_masks.append(texture > TRESH)  
                
                all_tasks_mask = np.all(task_masks, axis=0)

                conditions_array = np.zeros(task_masks[0].shape, dtype=int)
                overlap_array = np.zeros(task_masks[0].shape, dtype=int)

                conditions_array = np.sum(task_masks, axis=0)
                overlap_array[all_tasks_mask] = 1 #overlap map across all tasks

                cmap = ListedColormap(['#4793AF', '#FFC470', '#6C946F', '#DD5746'])

                # if 'confidence_neg' in contrast:
                #     conditions_array[pnas_mask] = 1  
                #     conditions_array[naconf_mask] = 2  
                #     conditions_array[encodeprob_mask] = 3  
                #     conditions_array[explore_mask] = 4  
                #     cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746', '#FAC67A'])
                # else:
                #     conditions_array[naconf_mask] = 1  
                #     conditions_array[explore_mask] = 2     
                #     conditions_array[encodeprob_mask] = 3  
                #     conditions_array[pnas_mask] = 4
                #     cmap = ListedColormap(['#FFC470', '#FAC67A', '#DD5746', '#4793AF'])          
                

                annot_fpath = os.path.join(ANNOT_DIR, annot_fname)
                hcpmmp1_annot = nib.freesurfer.io.read_annot(annot_fpath)
                annot_reference_labels = hcpmmp1_annot[0]
                annot_reference_names = [name.decode('UTF-8') for name in hcpmmp1_annot[2]]
                indices_overlap = np.where(overlap_array==1)[0]
                levels = np.unique([annot_reference_labels[i] for i in indices_overlap])
                names = np.unique([annot_reference_names[annot_reference_labels[i]] for i in indices_overlap])


                for view in ['lateral', 'medial']:

                    # #full glasser parcelation
                    # figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, hemi=hemi, vmin=1, vmax=4,view=view, alpha=None)
                    # plotting.plot_surf_contours(
                    #     surf_mesh=mesh,
                    #     roi_map=annot_reference_labels,
                    #     hemi=hemi,
                    #     figure=figure,
                    #     cmap='Pastel2',
                    #     legend=True)            
                    # fname = f'{contrast}_allOverlap_explore_surface_{hemi}_{view}_HCPMMP1.png'
                    # plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

                    #ROI with overlap only
                    figure = plotting.plot_surf_roi(surf_mesh=mesh,roi_map=conditions_array, bg_map=bg_map, cmap=cmap, vmin=1, vmax=4,hemi=hemi, view=view, alpha=None)
                    fname = f'{contrast}_allOverlap_explore_surface_{hemi}_{view}_.png'
                    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
                    fname = f'{contrast}_allOverlap_explore_surface_{hemi}_{view}.svg'
                    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")

                    plotting.plot_surf_contours(
                        surf_mesh=mesh,
                        roi_map=annot_reference_labels,
                        hemi=hemi,
                        levels=levels,
                        labels = names,
                        figure=figure,
                        cmap='Pastel2',
                        legend=True)            
                    fname = f'{contrast}_allOverlap_explore_surface_{hemi}_{view}_HCPMMP1_ROIs.png'
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

            plt.close('all')


##### Plot colorbars for overlap analysis ######
if CBAR_CLUSTER_OVERLAP_ALL:
    categories = [0, 1, 2, 3]
    labels = ["no overlap", "2 overlap", "3 overlap", "all overlap"]

    cmap = cmap = ListedColormap(['#4793AF', '#FFC470',  '#6C946F', '#DD5746'])
    norm = BoundaryNorm(categories + [len(categories)], cmap.N)  # Define boundaries for the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.5, 5))  # Adjust size as needed
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    tick_positions = [0.5, 1.5, 2.5, 3.5]  
    cbar.set_ticks(tick_positions)
    cbar.ax.set_yticklabels(labels)  
    cbar.ax.tick_params(labelsize=24)  
    fname = 'colorbar_overlap_4cat.png'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)
    fname = 'colorbar_overlap_4cat.svg'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", transparent=True)
    plt.close('all')

if CBAR_CLUSTER_OVERLAP_PROB:
    categories = [0, 1, 2]
    labels = ["no overlap", "2 overlap", "all overlap"]

    cmap = cmap = ListedColormap(['#4793AF', '#FFC470', '#DD5746'])
    norm = BoundaryNorm(categories + [len(categories)], cmap.N)  
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.5, 5))  
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    tick_positions = [0.5, 1.5, 2.5]  
    cbar.set_ticks(tick_positions)
    cbar.ax.set_yticklabels(labels)  

    cbar.ax.tick_params(labelsize=10)  
    fname = 'colorbar_overlap_3cat-png'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)
    plt.close('all')


##### Plot group level correleation heatmap ######
if CORR_ALL:
    plt.rcParams.update({'font.size': 18})

    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
    cmap = np.genfromtxt('/home/ah278717/hansen_receptors/data/colourmap.csv', delimiter=',')
    cmap_div = ListedColormap(cmap)
    cmap_seq = ListedColormap(cmap[128:, :])

    # Initialize a list to store all data
    data_list = []
    for task in tasks:  # Loop through task folders
        if task == 'Explore':
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
        else:
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')

        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        for contrast_file in os.listdir(task_path):  # Loop through subject files
            contrast_path = os.path.join(task_path, contrast_file)
            if contrast_file.endswith(f"group_confidence_schaefer_effect_size{add_info}.pickle"):  
                confidence_array = np.load(contrast_path, allow_pickle=True).flatten()
                data_list.append({'task': task, 'contrast': 'confidence', 'data': confidence_array})
            elif contrast_file.endswith(f"group_surprise_schaefer_effect_size{add_info}.pickle"):  # Load surprise arrays
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
            mask = (data_i != 0) & (data_j != 0) #to remove the previous NAN from NAConf
            # Apply the mask
            filtered_data_i = data_i[mask]
            filtered_data_j = data_j[mask]
            correlation_matrix[i, j] = pearsonr(filtered_data_i, filtered_data_j)[0] 

    # Create a mapping of indices to (task, subject, contrast)
    labels = [f"{row['task']}_{row['contrast']}" for _, row in all_data.iterrows()]

    correlation_df = pd.DataFrame(correlation_matrix, index=labels, columns=labels)

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
    fname = '4group_correlation_heatmap_by_task.png'
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
    mask = np.triu(np.ones_like(correlation_df_reordered, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df_reordered, annot=True, fmt=".2f", cmap=cmap_div, square=True, center=0)
    fname = '4group_correlation_heatmap_by_contrast.png'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=300)
    plt.close('all')


#### correlation with just the probabaility learning datasets
if CORR_PROB:
    tasks = ['EncodeProb', 'NAConf', 'PNAS']  # Excluding 'Explore'
    cmap = np.genfromtxt('/home/ah278717/hansen_receptors/data/colourmap.csv', delimiter=',')
    cmap_div = ListedColormap(cmap)

    # Load data
    data_list = []
    for task in tasks:
        if task == 'Explore':
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
        else:
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                    
        if task in ['NAConf']:
            if params.remove_trials:
                add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        for contrast_file in os.listdir(task_path):
            contrast_path = os.path.join(task_path, contrast_file)
            if contrast_file.endswith(f"group_confidence_schaefer_effect_size{add_info}.pickle"):
                confidence_array = np.load(contrast_path, allow_pickle=True).flatten()
                data_list.append({'task': task, 'contrast': 'confidence', 'data': confidence_array})
            elif contrast_file.endswith(f"group_surprise_schaefer_effect_size{add_info}.pickle"):
                surprise_array = np.load(contrast_path, allow_pickle=True).flatten()
                data_list.append({'task': task, 'contrast': 'surprise', 'data': surprise_array})

    all_data = pd.DataFrame(data_list)

    # Calculate correlation matrix
    n = len(all_data)
    correlation_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            data_i = all_data.iloc[i]['data']
            data_j = all_data.iloc[j]['data']
            mask = (data_i != 0) & (data_j != 0) #to remove the previous NAN from NAConf
            # Apply the mask
            filtered_data_i = data_i[mask]
            filtered_data_j = data_j[mask]
            correlation_matrix[i, j] = pearsonr(filtered_data_i, filtered_data_j)[0] 

    # Create labels and save first heatmap grouped by task
    labels = [f"{row['task']}_{row['contrast']}" for _, row in all_data.iterrows()]
    correlation_df = pd.DataFrame(correlation_matrix, index=labels, columns=labels)

    # Group by task
    grouped_by_task = all_data.sort_values(by='task')
    task_labels = [f"{row['task']}_{row['contrast']}" for _, row in grouped_by_task.iterrows()]
    task_indices = [labels.index(label) for label in task_labels]
    correlation_matrix_by_task = correlation_matrix[np.ix_(task_indices, task_indices)]
    correlation_df_by_task = pd.DataFrame(
        correlation_matrix_by_task, index=task_labels, columns=task_labels
    )
    correlation_df_by_task.to_csv(os.path.join(output_dir, 'correlation_df_by_task.csv'))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df_by_task, annot=True, fmt=".2f", cmap=cmap_div, square=True, center=0)
    plt.savefig(os.path.join(output_dir, '3group_correlation_heatmap_by_task.png'), bbox_inches="tight", dpi=300)

    # Group by contrast
    grouped_by_contrast = all_data.sort_values(by='contrast')
    contrast_labels = [f"{row['task']}_{row['contrast']}" for _, row in grouped_by_contrast.iterrows()]
    contrast_indices = [labels.index(label) for label in contrast_labels]
    correlation_matrix_by_contrast = correlation_matrix[np.ix_(contrast_indices, contrast_indices)]
    correlation_df_by_contrast = pd.DataFrame(
        correlation_matrix_by_contrast, index=contrast_labels, columns=contrast_labels
    )
    correlation_df_by_contrast.to_csv(os.path.join(output_dir, 'correlation_df_by_contrast.csv'))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df_by_contrast, annot=True, fmt=".2f", cmap=cmap_div, square=True, center=0)
    plt.savefig(os.path.join(output_dir, '3group_correlation_heatmap_by_contrast.png'),bbox_inches="tight", dpi=300)

    plt.close('all')