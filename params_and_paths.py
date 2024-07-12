#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
DB_NAME = 'EncodeProb' #this will have to determine all the following parameters in a flexible script
# other options: 'EncodeProb','NAConf', 'Explore', 'PNAS'
# MASK = 'mask_GreyMatter_0_25_WithoutCereb.nii'
# MASK_NAME = 'GrayMatter_noCereb'
MASK_NAME = 'harvard_oxford_cortical' #'harvard_oxford_cortical'; harvard_oxford_subcortical; schaefer
RECEPTOR_SOURCE = 'PET' #,'PET' or 'autorad_zilles44'
#---------------------------------------------------

class Params:
    def __init__(self, db=DB_NAME, mask = MASK_NAME):
        self.db = db
        self.mask = mask

        if mask == 'schaefer':
            self.parcelated = True
        else:
            self.parcelated = False

        self.hpf = 1/128
        self.hrf = 'spm'

        if db != 'Explore':
            self.io_options = {'p_c': 1/75, 'resol': 20} 
        if db not in ['Explore', 'PNAS']:
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = None #no smoothing necessary because of 7T fMRI
        elif db == 'PNAS':
            self.seq_type = 'transition'
            self.smoothing_fwhm = 5

        self.latent_vars = ['surprise', 'confidence', 'predictability', 'predictions']

        #participants to ignore and session deviations
        if db == 'NAConf':
            self.ignore = [3, 6, 9, 51, 54, 59]
            self.session = []
        elif db == 'EncodeProb':
            self.ignore = [1, 4, 12, 20]
            self.session = {6:[1,3,4,5], 20:[1,2,3,5], 21:[1,2,3,5]}
        elif db == 'Explore':
            self.ignore = [9, 17, 46]
            self.session = []
        elif db == 'PNAS':
            self.ignore = []
            self.session = []

        # subnums_explore = {4: 6,
#                    6: 4,
#                    25: 28,
#                    28: 25}

        #mask details
        if mask == 'schaefer':
            self.mask_details = '100' #number of regions


class Paths:
    def __init__(self, db=DB_NAME):

        if db == 'NAConf':
            self.data_dir = 'MeynielMazancieux_NACONF_prob_2021'
        elif db == 'EncodeProb':
            self.data_dir = 'EncodeProb_BounmyMeyniel_2020'
        elif db == 'Explore':
            self.data_dir = '/home_local/EXPLORE/DATA'
        elif db == 'PNAS':
            self.data_dir = 'Meyniel_MarkovGuess_2014'

        self.home_dir = '/home_local/alice_hodapp/NeuroModAssay'
        self.root_dir = '/neurospin/unicog/protocols/IRMf'


class Receptors:
    def __init__(self, source=RECEPTOR_SOURCE):
        self.source = RECEPTOR_SOURCE

        if source == 'PET':
            self.receptor_names = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                                "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                                "MOR", "NET", "NMDA", "VAChT"]
            self.serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
            self.acetylcholine = ["A4B2", "M1", "VAChT"]
            self.noradrenaline = ["NET"]
            self.opioid = ["MOR"]
            self.glutamate = ["mGluR5"]
            self.histamine = ["H3"]
            self.gaba = ["GABAa"]
            self.dopamine = ["D1", "D2", "DAT"]
            self.cannabinnoid = ["CB1"]
            self.exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
            self.inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']

        if source == 'autorad_zilles44':
            self.receptor_names = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa/BZ', 'GABAb', 'm1', 'm2', 'm3', 'a4b2',
                                'a1', 'a2', '5-HT1a', '5-HT2', 'D1'] 
            self.serotonin = ['5-HT1a', '5-HT2']
            self.acetylcholine = ['m1', 'm2', 'm3', 'a4b2']
            self.noradrenaline = ['a1', 'a2']
            self.glutamate = ['AMPA', 'NMDA', 'kainate']
            self.gaba = ['GABAa', 'GABAa/BZ', 'GABAb']
            self.dopamine = ['D1']
            self.exc = ['AMPA', 'NMDA', 'kainate', 'm1', 'm3', 'a4b2', 'a1', '5-HT2', 'D1']
            self.inh = ['GABAa', 'GABAa/BZ', 'GABAb', 'm2', 'a2', '5-HT1a']

