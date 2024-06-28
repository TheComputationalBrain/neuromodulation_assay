#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:18:09 2024

@author: Alice Hodapp

This loads the cortical recepto densities originally published in Zilles & Palomero-Gallagher (2017), made available here: https://github.com/AlGoulas/receptor_principles,
and then manually maps the areas to different brain atlasses.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from params_and_paths import * 


receptor_path = '/home/ah278717/hansen_receptors/data/autoradiography/' #path to downloaded data from Hansen et al. (2022)
output_dir = os.path.join(home_dir[DATA_ACCESS],'receptors', 'autorad_zilles44')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

autorad_zilles44 = np.load(os.path.join(receptor_path, 'ReceptData.npy'))
receptor_names = np.load(os.path.join(receptor_path, 'ReceptorNames.npy'))


autorad_schaefer100 = np.zeros((50, autorad_zilles44.shape[1]))
# mapping between zilles and schaefer (done manually)
zilles44_to_schaefer100 = \
    [[21],  # area '36' // 0
        [9],   # area 'V3v' // 1
        [6],   # area 'V2v' // 2
        [4, 6],   # area 'V1', 'V2v'  // 3
        [4, 5, 6],   # area 'V1', 'V2v', 'V2d' // 4
        [4],   # area 'V1' // 5
        [np.nan],  # // 6
        [8, 7],   # area 'V3d', 'V3a' // 7
        [8],   # area 'V3d' // 8
        [17],  # area '42' // 9
        [16],  # area '41' // 10
        [np.nan], # // 11
        [0, 3, 2],  # area '1', '3b', '3a' \\ 12
        [0, 3, 2],  # area '1', '3b', '3a' \\ 13
        [0, 3, 2],  # area '1', '3b', '3a'\\ 14
        [14],  # area '37L' \\ 15
        [27, 1],  # area 'PFt', '2' \\ 16
        [np.nan],  # // 17
        [1],  # area '2' // 18
        [24],  # area '5M' // 19
        [23],  # area '5L' // 20
        [0, 34],  # area '1', '6' // 21
        [35],  # area '8' // 22
        [28, 27],  # area 'PFm', 'PFt' // 23
        [np.nan],  # insula // 24
        [41],  # area '45' // 25
        [36],  # area '9' // 26
        [29],  # area '24' // 27
        [24],  # area '5M' // 28
        [35],  # area '8' // 29
        [39],  # area '11' // 30
        [22],  # area '38' // 31
        [18],  # area '20' // 32
        [28, 27],  # area 'PFm' 'PFt' // 33
        [42],  # area '46' // 34
        [7],  # area 'V3A' // 35
        [31],  # area '23' (and 31?) // 36
        [20, 19],  # area '22', '21' // 37
        [19, 20],  # area '21', '22' // 38
        [19, 20],  # area '21', '22' // 39
        [25, 26],  # area 'PGa', 'PGp' // 40
        [43],  # area '47' // 41
        [41, 43, 42],  # area '45', '47', '46' // 42
        [29, 30, 38],  # area '24', '32', '10M' // 43
        [37],  # area '10L' // 44
        [36],  # area '9' // 45
        [35, 36],  # area '8', '9' // 46
        [35],  # area '8' // 47
        [32],  # area '31' // 48
        [32]]  # area '31' // 49 

for n in range(autorad_schaefer100.shape[0]):
    if np.isnan(zilles44_to_schaefer100[n][0]):
        autorad_schaefer100[n, :] = np.nan
    elif len(zilles44_to_schaefer100[n]) == 1:
        autorad_schaefer100[n, :] = autorad_zilles44[zilles44_to_schaefer100[n], :]
    elif len(zilles44_to_schaefer100[n]) > 1:
        autorad_schaefer100[n, :] = np.mean(autorad_zilles44[zilles44_to_schaefer100[n], :], axis=0).T

with open(os.path.join(output_dir, 
                        'receptor_density_schaefer_100.pickle'), 'wb') as f:
            pickle.dump(autorad_schaefer100, f)

### plotting 
plot_path = os.path.join(output_dir,'figures') 
if not os.path.exists(plot_path):
        os.makedirs(plot_path) 

serotonin = ['5-HT1a', '5-HT2']
acetylcholine = ['m1', 'm2', 'm3', 'a4b2']
noradrenaline = ['a1', 'a2']
glutamate = ['AMPA', 'NMDA', 'kainate']
gaba = ['GABAa', 'GABAa/BZ', 'GABAb']
dopamine = ['D1']
receptor_groups = [serotonin, acetylcholine, noradrenaline, glutamate, gaba, dopamine]

cmap = np.genfromtxt('../hansen_receptors/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

#receptor_data = np.load(os.path.join(output_dir, f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True)
ordered_receptors = [receptor for group in receptor_groups for receptor in group]
df = pd.DataFrame(zscore(autorad_schaefer100, nan_policy='omit'), columns=receptor_names)
df = df[ordered_receptors]
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap=cmap_div, vmin=-1, vmax=1, linewidths=0.6, square=True)
plt.title(f'Autoradiography dataset Zilles44 (Schaefer 100): Correlation of Receptors')
current_pos = 0
for group in receptor_groups:
    group_size = len(group)
    plt.gca().add_patch(Rectangle((current_pos, current_pos), group_size, group_size, fill=False, edgecolor='black', lw=2))
    current_pos += group_size

fig_fname = f'receptor_corr_matrix_schaefer_100.png'
plt.savefig(os.path.join(plot_path, fig_fname))




