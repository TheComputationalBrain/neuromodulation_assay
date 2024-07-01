#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
DB_NAME = 'EncodeProb' #this will have to determine all the following parameters in a flexible script
# other options: 'NAConf', 'Explore', 'PNAS'
DATA_ACCESS = 'server' # 'private' for testing; otherwise: 'server' 
#mask = './masks/spm12/tpm/mask_ICV.nii'
# MASK = 'mask_GreyMatter_0_25_WithoutCereb.nii'
# MASK_NAME = 'GrayMatter_noCereb'
HPF    =  1/128
SMOOTHING_FWHM = None
HRF = 'spm'
RES = 20 #resolution for the IO hmm
MASK_NAME = 'schaefer' #'harvard_oxford_cortical'; harvard_oxford_subcortical
RECEPTOR_SOURCE = 'autorad_zilles44' #,'PET'
PARCELATED = True

#----------------------------------------------------

mask_details = {'schaefer': '100' #number of regions: from 100 to 1000 in steps of 100
                }

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
          'EncodeProb': {6:[1,3,4,5], 20:[1,2,3,5], 21:[1,2,3,5]},
          'Explore': [],
          'PNAS': []}

home_dir = {'private': '/Volumes/NEUROSPIN/Neuromod',
            'server': '/home_local/alice_hodapp/NeuroModAssay'} 

root_dir = {'private': '/Volumes/NEUROSPIN',
            'server': '/neurospin/unicog/protocols/IRMf'}

pet = {
    'receptor_names': ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                       "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                       "MOR", "NET", "NMDA", "VAChT"], 
    'serotonin': ['5-HT1a', '5-HT2'],
    'acetylcholine': ['m1', 'm2', 'm3', 'a4b2'],
    'noradrenaline': ['a1', 'a2'],
    'glutamate': ['AMPA', 'NMDA', 'kainate'],
    'gaba': ['GABAa', 'GABAa/BZ', 'GABAb'],
    'dopamine': ['D1'],
    'exc': ['AMPA', 'NMDA', 'kainate', 'm1', 'm3', 'a4b2', 'a1', '5-HT2', 'D1'],
    'inh': ['GABAa', 'GABAa/BZ', 'GABAb', 'm2', 'a2', '5-HT1a']
}

autorad_zilles44 = {
    'receptor_names': ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa/BZ', 'GABAb', 'm1', 'm2', 'm3', 'a4b2',
                       'a1', 'a2', '5-HT1a', '5-HT2', 'D1'],
    'serotonin': ['5-HT1a', '5-HT2'],
    'acetylcholine': ['m1', 'm2', 'm3', 'a4b2'],
    'noradrenaline': ['a1', 'a2'],
    'glutamate': ['AMPA', 'NMDA', 'kainate'],
    'gaba': ['GABAa', 'GABAa/BZ', 'GABAb'],
    'dopamine': ['D1'],
    'exc': ['AMPA', 'NMDA', 'kainate', 'm1', 'm3', 'a4b2', 'a1', '5-HT2', 'D1'],
    'inh': ['GABAa', 'GABAa/BZ', 'GABAb', 'm2', 'a2', '5-HT1a']
}

receptors = {
    'PET': pet,
    'autorad_zilles44': autorad_zilles44
}