# this meta level script calls all functions from other scripts to plot the figures that are used in the paper


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path

from invariance_effect_maps import plot_cluster_overlap_all, plot_colorbar_overlap, plot_correlations
from variance_explained import plot_variance_explained, plot_explained_variance_ratio
from receptor_effect_map_relationship import load_dominance_data, aggregate_dominance, plot_dominance_bars, plot_dominance_heatmap, plot_legend_dominance_bars, plot_separate_colorbar

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils.main_funcs as mf
from config.loader import load_config

#default settings
params, paths, rec = load_config('all', return_what='all')

# folder for figures only
OUTPUT_DIR = os.path.join(paths.home_dir, 'figures_quad')

#invariance of functional activity: cluster overlap and colorbar
mf.set_publication_style(font_size=7, layout="3-across")
plot_cluster_overlap_all(params.tasks, ['surprise', 'confidence', 'surprise_neg', 'confidence_neg'], OUTPUT_DIR)
fig,ax = plot_colorbar_overlap()
mf.save_figure(fig, OUTPUT_DIR, 'colorbar_overlap_4cat')

#invariance of functional activity: correlations between maps
cmap_div = mf.get_custom_colormap('diverging')
plot_correlations(["cross", "lower_triangles"], cmap_div,  OUTPUT_DIR)

#variance explained: R2 and null model / ratio of explained variance
fig, ax = plot_variance_explained(params, legend=True)
mf.save_figure(fig, OUTPUT_DIR, f"barplot_explained_variance")
fig, ax = plot_explained_variance_ratio(params, legend=False)
mf.save_figure(fig, OUTPUT_DIR, f"barplot_explained_variance_ratio")

mf.set_publication_style(font_size=7, layout="2-across")

for latent_var in params.latent_vars:
    #dominance analysis and legend 
    results = load_dominance_data(params.tasks, latent_var, model_type='linear') # or 'lin+quad' for supplementary plot
    combined, per_study_means = aggregate_dominance(results)

    fig, ax = plot_dominance_bars(combined, rec.receptor_groups, rec.receptor_class, rec.receptor_label_formatted, title=latent_var)
    mf.save_figure(fig, OUTPUT_DIR, f"group_{latent_var}_dominance")

    cmap_pos = mf.get_custom_colormap('pos')
    fig, ax = plot_dominance_heatmap(per_study_means, rec.receptor_groups, cmap_pos, rec.receptor_label_formatted, rename_tasks=True, params=params)
    mf.save_figure(fig, OUTPUT_DIR, f"heatmap_{latent_var}")

fig, ax = plot_separate_colorbar(cmap=cmap_pos, vmin=0, vmax=0.18)
mf.save_figure(fig, OUTPUT_DIR, f"colorbar_heatmap")

fig = plot_legend_dominance_bars(rec, ncol=6, fig_width=7, fig_height=1.5)
mf.save_figure(fig, OUTPUT_DIR, f"legend_dominance_bar")