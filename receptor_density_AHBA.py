#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:39:40 2024

This creates a gene expresion levels matrix (as proxy of receptor density) from the Allen human brain atlas microarray dataset. 
It follows the reccomendations by Arnatkevic̆iūtė et al., 2019, NeuroImage.
"""

import os
import numpy as np
import pandas as pd
import abagen
from nilearn.datasets import fetch_atlas_schaefer_2018
from params_and_paths import Paths, Params

paths = Paths()
params = Params()

ATLAS_SCHAEFER = False
ATLAS_DESIKAN = True

output_dir = os.path.join(paths.home_dir,'receptors', 'AHBA')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

if ATLAS_SCHAEFER:
    atlas = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2) 
    atlas.labels = np.insert(atlas.labels, 0, "background")
    #create csv with info on atlas
    id_column = list(np.arange(1, 101))
    hemisphere_column = ['R'] * 50 + ['L'] * 50
    structure_column = ['cortex'] * 100
    atlas_info = pd.DataFrame({
        'id': id_column,
        'hemisphere': hemisphere_column,
        'structure': structure_column
    })
    expression = abagen.get_expression_data(atlas['maps'], atlas_info, lr_mirror='bidirectional', missing='interpolate')  

if ATLAS_DESIKAN:
   atlas = abagen.fetch_desikan_killiany()   
   expression = abagen.get_expression_data(atlas['image'], atlas['info'], missing='interpolate')

#gene expression of interest (based on other literature)
genes_NA = ['ADRA1A', 'ADRA1B', 'ADRA1D', 'ADRA2A', 'ADRA2C', 'ADRB1', 'ADRB2']
genes_ST = ['HTR1A', 'HTR1E', 'HTR2A', 'HTR3B', 'HTR3C', 'HTR4','HTR7'] #HTR1D, HTR2A, 'HTR2C'
genes_ACH = ['CHRM1', 'CHRM2', 'CHRM4', 'CHRNB2']
genes_DA = ['DRD1', 'DRD2', 'DRD4']

#save all
expression.to_csv(os.path.join(output_dir,'gene_expression_all_desikan.csv'), index=False)

#save genes that are of interest 
expression_NA = expression[genes_NA]
expression_NA.to_csv(os.path.join(output_dir,'gene_expression_NA_desikan.csv'), index=False)

genes_full = genes_NA + genes_ST + genes_ACH + genes_DA
expression_full = expression[genes_full].copy()
columns_to_average = [col for col in expression_full.columns if col.startswith('HTR3')] #average the 5-HT3 genes that all express the same receptor
mean_5HR3 = expression_full.loc[:, columns_to_average].mean(axis=1)
expression_full['5HR3'] = mean_5HR3
expression_full = expression_full.drop(columns=columns_to_average)

expression_full.to_csv(os.path.join(output_dir,'gene_expression_complex_desikan.csv'), index=False)












