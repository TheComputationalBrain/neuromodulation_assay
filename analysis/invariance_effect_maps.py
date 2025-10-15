import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting, datasets, surface
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import pearsonr
from neuromaps import transforms
from params_and_paths import Paths, Params
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import main_funcs as mf


# --- Global configuration ---
paths = Paths()
params = Params()
RESOLUTION = 'fsaverage'
EXPLORE_MODEL = 'noEntropy_noER'
FWHM = 5
TRESH = -np.log10(0.05)
VMAX = -np.log10(1 / 100000)
OUTPUT_DIR = os.path.join(paths.home_dir, 'domain_general')
os.makedirs(OUTPUT_DIR, exist_ok=True)
fsaverage = datasets.fetch_surf_fsaverage(mesh=RESOLUTION)


# Plot individual significant clusters per dataset & contrast
def plot_individual_clusters(tasks, contrasts):
    for contrast in contrasts:
        cmap = 'Blues' if '_neg' in contrast else 'Reds'
        for task in tasks:
            _, add_info = mf.get_beta_dir_and_info(task)
            group_dir = (
                os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
                if task == 'Explore'
                else os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
            )

            img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
            img = nib.load(img_path)

            # Whole brain (both hemispheres)
            plotting.plot_img_on_surf(
                img, surf_mesh=RESOLUTION,
                hemispheres=['left', 'right'], views=['lateral', 'medial'],
                threshold=TRESH, vmax=VMAX, title=f"{task} {contrast}",
                colorbar=True, inflate=True, cmap=cmap, cbar_tick_format='%.2f'
            )
            out_path = os.path.join(OUTPUT_DIR, 'plots_by_task')
            os.makedirs(out_path, exist_ok=True)
            plt.savefig(os.path.join(out_path, f'{task}_{contrast}_cluster_mass.png'), dpi=300)
            plt.close('all')

            # Right hemisphere detailed
            texture = surface.vol_to_surf(img, fsaverage.pial_right)
            plotting.plot_surf_stat_map(
                fsaverage.infl_right, texture, bg_map=fsaverage.sulc_right,
                hemi='right', threshold=TRESH, vmax=VMAX, title=f"{task} {contrast}",
                colorbar=True, cmap=cmap, cbar_tick_format='%.2f'
            )
            plt.savefig(os.path.join(out_path, f'{task}_{contrast}_cluster_mass_right.png'), dpi=300)
            plt.close('all')


# Plot overlap across all datasets
def plot_cluster_overlap_all(tasks, contrasts):
    with open(os.path.join(OUTPUT_DIR, 'ROIs_overlap_explore.txt'), "w") as outfile:
        for contrast in contrasts:
            rois_with_overlap = []

            for hemi in ['right', 'left']:
                task_masks = []
                for task in tasks:
                    _, add_info = mf.get_beta_dir_and_info(task)
                    group_dir = (
                        os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
                        if task == 'Explore'
                        else os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                    )
                    img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
                    img = nib.load(img_path)
                    texture = surface.vol_to_surf(img, getattr(fsaverage, f'pial_{hemi}'))
                    task_masks.append(texture > TRESH)

                # compute overlap
                all_tasks_mask = np.all(task_masks, axis=0)
                conditions_array = np.sum(task_masks, axis=0)

                cmap = ListedColormap(['#4793AF', '#FFC470', '#6C946F', '#DD5746'])

                # Save plots
                for view in ['lateral', 'medial']:
                    mesh = getattr(fsaverage, f'infl_{hemi}')
                    bg_map = getattr(fsaverage, f'sulc_{hemi}')
                    figure = plotting.plot_surf_roi(
                        surf_mesh=mesh, roi_map=conditions_array, bg_map=bg_map,
                        cmap=cmap, vmin=1, vmax=4, hemi=hemi, view=view
                    )
                    base = f'{contrast}_allOverlap_explore_{hemi}_{view}'
                    plt.savefig(os.path.join(OUTPUT_DIR, f'{base}.png'), bbox_inches="tight", dpi=300)
                    plt.savefig(os.path.join(OUTPUT_DIR, f'{base}.svg'), bbox_inches="tight")


# Plot colorbar legend for overlap
def plot_colorbar_overlap():
    categories = [0, 1, 2, 3]
    labels = ["no overlap", "2 overlap", "3 overlap", "all overlap"]
    cmap = ListedColormap(['#4793AF', '#FFC470', '#6C946F', '#DD5746'])
    norm = BoundaryNorm(categories + [len(categories)], cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(0.5, 5))
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(labelsize=24)

    for ext in ['png', 'svg']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'colorbar_overlap_4cat.{ext}'), bbox_inches="tight", dpi=300)
    plt.close('all')


# Group-level correlation analysis (modular)
def plot_correlations(plots_to_generate=["all"]):
    """
    Generate correlation plots between second-level maps across tasks and contrasts.

    plots_to_generate: list of strings controlling which plots are created.
        Options include:
        - "all": full correlation heatmap (default)
        - "contrast_sorted": heatmap sorted by contrast/task
        - "confidence": submatrix for confidence contrasts
        - "surprise": submatrix for surprise contrasts
        - "cross": cross correlation between confidence and surprise
        - "lower_triangles": masked lower-triangle plots for publication-style visuals
    """
    plt.rcParams.update({'font.size': 16})

    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
    cmap = np.genfromtxt('/home/ah278717/hansen_receptors/data/colourmap.csv', delimiter=',')
    cmap_div = ListedColormap(cmap)

    # --- Gather data ---
    data_list = []
    for task in tasks:
        _, add_info = mf.get_beta_dir_and_info(task)
        base_dir = (
            os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
            if task == 'Explore'
            else os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
        )
        for fname in os.listdir(base_dir):
            for contrast in ['confidence', 'surprise']:
                if fname.endswith(f"{contrast}_schaefer_effect_map{add_info}.nii.gz"):
                    img = nib.load(os.path.join(base_dir, fname))
                    surf_data = transforms.mni152_to_fsaverage(img, fsavg_density='41k')
                    data = np.hstack([d.agg_data().T for d in surf_data])
                    data_list.append({'task': task, 'contrast': contrast, 'data': data})

    df = pd.DataFrame(data_list)
    n = len(df)
    corr_matrix = np.zeros((n, n))

    # --- Compute correlations ---
    for i in range(n):
        for j in range(n):
            d1, d2 = df.iloc[i]['data'], df.iloc[j]['data']
            mask = (d1 != 0) & (d2 != 0)
            if np.any(mask):
                corr_matrix[i, j] = pearsonr(d1[mask], d2[mask])[0]
            else:
                corr_matrix[i, j] = np.nan

    labels = [f"{r.task}_{r.contrast}" for _, r in df.iterrows()]
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_df.csv'))

    # --- Helper for saving ---
    def save_heatmap(matrix, fname, title=None, mask=None, annot=False):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=annot, fmt=".2f", cmap=cmap_div,
                    center=0, square=True, mask=mask, cbar_kws={'label': 'r'})
        if title:
            plt.title(title)
        plt.tight_layout()
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'), dpi=300, bbox_inches='tight')
        plt.close('all')

    # --- 1. Full heatmap ---
    if "all" in plots_to_generate:
        save_heatmap(corr_df, 'correlation_heatmap_all', 'All contrasts & tasks', annot=True)

    # --- 2. Contrast-sorted ---
    if "contrast_sorted" in plots_to_generate:
        order = sorted(labels, key=lambda x: ('confidence' not in x, x))
        corr_sorted = corr_df.loc[order, order]
        save_heatmap(corr_sorted, 'correlation_heatmap_contrast_sorted', 'Contrast-sorted correlations')

    # --- 3. Confidence-only ---
    if "confidence" in plots_to_generate:
        conf_labels = [l for l in labels if "confidence" in l]
        conf_df = corr_df.loc[conf_labels, conf_labels]
        save_heatmap(conf_df, 'correlation_confidence_only', 'Confidence contrasts')

    # --- 4. Surprise-only ---
    if "surprise" in plots_to_generate:
        surpr_labels = [l for l in labels if "surprise" in l]
        surpr_df = corr_df.loc[surpr_labels, surpr_labels]
        save_heatmap(surpr_df, 'correlation_surprise_only', 'Surprise contrasts')

    # --- 5. Cross correlation between confidence and surprise ---
    if "cross" in plots_to_generate:
        conf_labels = [l for l in labels if "confidence" in l]
        surpr_labels = [l for l in labels if "surprise" in l]
        cross_df = corr_df.loc[conf_labels, surpr_labels]
        save_heatmap(cross_df, 'correlation_conf_surprise_cross', 'Cross correlation: confidence vs surprise')

    # --- 6. Lower-triangle versions (cleaner visuals) ---
    if "lower_triangles" in plots_to_generate:
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        save_heatmap(corr_df, 'correlation_lower_triangle', 'Lower triangle only', mask=mask)

        conf_labels = [l for l in labels if "confidence" in l]
        surpr_labels = [l for l in labels if "surprise" in l]

        conf_df = corr_df.loc[conf_labels, conf_labels]
        mask_conf = np.triu(np.ones_like(conf_df, dtype=bool))
        save_heatmap(conf_df, 'correlation_confidence_lower_triangle', 'Confidence (lower triangle)', mask=mask_conf)

        surpr_df = corr_df.loc[surpr_labels, surpr_labels]
        mask_surpr = np.triu(np.ones_like(surpr_df, dtype=bool))
        save_heatmap(surpr_df, 'correlation_surprise_lower_triangle', 'Surprise (lower triangle)', mask=mask_surpr)

# Main controller 
def run_analysis(
    run_individual=False,
    run_overlap=False,
    run_colorbar=False,
    run_correlations=False,
    corr_plots=None):
    """wrapper to run selected analyses."""
    tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']
    contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

    if run_individual:
        plot_individual_clusters(tasks, contrasts)

    if run_overlap:
        plot_cluster_overlap_all(tasks, contrasts)

    if run_colorbar:
        plot_colorbar_overlap()

    if run_correlations:
        plot_correlations(plots_to_generate=corr_plots or ["lower_triangles"])

if __name__ == "__main__":

    run_analysis(run_individual=True, run_overlap=True ,run_colorbar=True, run_correlations=True)
