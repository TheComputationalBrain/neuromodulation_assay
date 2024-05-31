#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp, Maëva L'Hôtellier
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
DB_NAME = 'EncodeProb' #this will have to determine all the following parameters in a flexible script
# other options: 'NAConf', 'Explore', 'PNAS'
DATA_ACCESS = 'server' # 'private' for testing; otherwise: 'server' 
#mask = './masks/spm12/tpm/mask_ICV.nii'
MASK = 'mask_GreyMatter_0_25_WithoutCereb.nii'
MASK_NAME = 'GrayMatter_noCereb'
HPF    =  1/128
SMOOTHING_FWHM = None
HRF = 'spm'
RES = 20 #resolution for the IO hmm
# Set Project Name
SAVE_DMTX_PLOT = True
#----------------------------------------------------

data_dir = {'NAConf': 'MeynielMazancieux_NACONF_prob_2021',
            'EncodeProb': 'EncodeProb_BounmyMeyniel_2020',
            'Explore': '/home_local/EXPLORE/DATA',
            'PNAS': 'Meyniel_MarkovGuess_2014'} 

# subnums_explore = {4: 6,
#                    6: 4,
#                    25: 28,
#                    28: 25}

ignore = {'NAConf': [3, 6, 9, 51, 54, 59],
          'EncodeProb': [1, 4, 12, 20],
          'Explore': [9, 17, 46],
          'PNAS': []}

sessions = {'NAConf': [],
          'EncodeProb': {6:[1,3,4,5], 21:[1,2,3,5]},
          'Explore': [],
          'PNAS': []}

home_dir = {'private': '/Volumes/NEUROSPIN/Neuromod',
            'server': '/home_local/alice_hodapp/NeuroModAssay'} 

root_dir = {'private': '/Volumes/NEUROSPIN',
            'server': '/neurospin/unicog/protocols/IRMf'}

mask_dir = {'private': '/Volumes/NEUROSPIN/SPM_masks',
            'server': '/neurospin/unicog/protocols/IRMf/Meyniel_Atlas_2022/SPM'}


