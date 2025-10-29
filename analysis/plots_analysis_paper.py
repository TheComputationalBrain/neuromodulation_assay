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
from receptor_effect_map_relationship import load_dominance_data, aggregate_dominance, plot_dominance_bars, plot_dominance_heatmap, plot_explore_dominance_heatmap

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors


paths = Paths(task = 'all')
params = Params(task = 'all')
rec = Receptors()


# folder for figures only
OUTPUT_DIR = os.path.join(paths.home_dir, 'figures')

#invariance of functional activity: cluster overlap and colorbar
mf.set_publication_style(font_size=7, layout="3-across")
# plot_cluster_overlap_all(params.tasks, ['surprise', 'confidence', 'surprise_neg', 'confidence_neg'], OUTPUT_DIR)
# fig,ax = plot_colorbar_overlap()
# mf.save_figure(fig, OUTPUT_DIR, 'colorbar_overlap_4cat')

#invariance of functional activity: correlations between maps
cmap_div = mf.get_custom_colormap('diverging')
plot_correlations(["lower_triangles"], cmap_div,  OUTPUT_DIR)

#variance explained: R2 and null model / ratio of explained variance
mf.set_publication_style(font_size=7, layout="2-across")
fig, ax = plot_variance_explained(params)
mf.save_figure(fig, OUTPUT_DIR, f"barplot_explained_variance")
fig, ax = plot_explained_variance_ratio(params)
mf.save_figure(fig, OUTPUT_DIR, f"barplot_explained_variance_ratio")

for latent_var in params.latent_vars:
    mf.set_publication_style(font_size=7, layout="2-across")
    #dominance analysis and legend 
    results = load_dominance_data(params.tasks, latent_var, model_type='linear')
    combined, per_study_means = aggregate_dominance(results, exclude_explore=True)

    fig, ax = plot_dominance_bars(combined, rec.receptor_groups, rec.receptor_class, rec.receptor_label_formatted)
    mf.save_figure(fig, OUTPUT_DIR, f"group_{latent_var}_dominance")

    cmap_pos = mf.get_custom_colormap('pos')
    fig, ax = plot_dominance_heatmap(per_study_means, rec.receptor_groups, cmap_pos, rec.receptor_label_formatted, rename_tasks=True)
    mf.save_figure(fig, OUTPUT_DIR, f"heatmap_{latent_var}")

fig, ax = plot_explore_dominance_heatmap(
params.latent_vars,
rec.receptor_groups,
rec.receptor_label_formatted,
cmap_pos)
mf.save_figure(fig, OUTPUT_DIR, f"Explore_heatmap")