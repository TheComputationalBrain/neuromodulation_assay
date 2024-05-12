#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp, Maëva L'Hôtellier
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
    
data_access = 'private' # 'private' for testing; 'server' or 'mounted' if using sshfs
model = 'basic'
#mask = './masks/spm12/tpm/mask_ICV.nii'
mask = 'mask_GreyMatter_0_25_WithoutCereb.nii'
mask_name = 'GrayMatter_noCereb'
HPF    =  1/128
smoothing_fwhm = None
HRF = 'spm'
res = 20 #resolution for the IO hmm
n_sessions = 4 #! just for testing 
#----------------------------------------------------

data_dir = {'NAConf': 'MeynielMazancieux_NACONF_prob_2021',
            'EncodeProb': 'EncodeProb_BounmyMeyniel_2020',
            'Explore': '/home_local/EXPLORE/DATA',
            'PNAS': 'Meyniel_MarkovGuess_2014'} 

mask_dir = {'private': '/Volumes/NEUROSPIN/SPM_masks',
            'mounted': '',
            'server': ''}

# subnums_explore = {4: 6,
#                    6: 4,
#                    25: 28,
#                    28: 25}

ignore = {'NAConf': [3, 6, 9, 51, 54, 59],
          'EncodeProb': [1, 4, 12, 20],
          'Explore': [9, 17, 46],
          'PNAS': []}

home_dir = {'private': '/Volumes/NEUROSPIN/Neuromod',
            'mounted': '/home/expe/beluga',
            'server': '/neurospin/unicog/protocols/IRMf/NeuroModAssay_HodappMeyniel_2024'}

root_dir = {'private': '/Volumes/NEUROSPIN',
            'mounted': '/home/expe/neurospin_local/unicog/protocols/IRMf',
            'server': '/neurospin/unicog/protocols/IRMf'}



