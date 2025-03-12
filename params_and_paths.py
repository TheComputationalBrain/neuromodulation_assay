#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
DB_NAME = 'PNAS' # other options: 'EncodeProb','NAConf', 'Explore', 'PNAS'
MASK_NAME =  'schaefer' #'spm_noCereb','harvard_oxford_cortical'; 'schaefer', 'desikan'
PARCELATED = False
RECEPTOR_SOURCE = 'PET2' #,'PET', 'PET2' or 'autorad_zilles44', 'AHBA', #PET2 is the dataset including alpha2
REDO_MASK = False #if False, the masking step will be skipped and the numpy datata array will be used instead except if smoothing/mask name have changed

#fixed at best setting:
UPDATE_REG = False #update or suprise + confidence as regressor
#---------------------------------------------------

class Params:
    def __init__(self, db=DB_NAME, mask = MASK_NAME, parcel = PARCELATED, update =UPDATE_REG, redo_mask=REDO_MASK):
        self.db = db
        self.mask = mask
        self.redo_mask = redo_mask

        self.parcelated = parcel
        self.update = update

        self.hpf = 1/128 
        self.hrf = 'spm'

        self.zscore_per_session = True 

        if db != 'Explore':
            self.io_options = {'p_c': 1/75, 'resol': 20} 

        if db == 'EncodeProb':
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = 5 #no smoothing necessary because of 7T fMRI -> just for comparison with other studies 
        elif db == 'NAConf':
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = 5
            self.naconf_behav_subj = \
            [1, 2, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
            43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 61] 
        elif db == 'PNAS':
            self.seq_type = 'transition'
            self.smoothing_fwhm = 5
        elif db == 'Explore':
            self.smoothing_fwhm = 5

        if UPDATE_REG:
            self.latent_vars = ['update', 'predictability', 'predictions']
        else:
            self.latent_vars = ['surprise','confidence', 'predictability', 'predictions']

        self.latent_vars_long = self.latent_vars + ['surprise_neg','confidence_neg', 'predictability_neg', 'predictions_neg']

        #participants to ignore and session deviations
        if db == 'NAConf':
            self.ignore = [3, 5, 6, 9, 36, 51]
            self.session = []
        elif db == 'EncodeProb':
            self.ignore = [1, 4, 12, 20]
            self.session = {6:[1,3,4,5], 20:[1,2,3,5], 21:[1,2,3,5]}
        elif db == 'Explore':
            self.ignore = [9, 17, 46]
            self.session = []
            #subject numbers are corrected in the formatted folder 
            # self.subnums_explore = {4: 6,
            #                         6: 4,
            #                         25: 28,
            #                         28: 25}
            self.split = False #split free and forced for the design matrix?
            self.reward = False 
            self.model = 'noEntropy_noER'   #'noEntropy',''noEntropy_reducedDM'; 'previously: US_reward'
            #self.io_variables = ['US', 'EU_chosen', 'ER_chosen', 'entropy_chosen'] 
            self.io_variables = ['US', 'EC_chosen'] 

        elif db == 'PNAS':
            self.ignore = []
            self.session = []

        #mask details
        if mask == 'schaefer':
            self.mask_details = '100' #number of regions
        elif mask == 'desikan':
            self.mask_details = '68'

class Paths:
    def __init__(self, db=DB_NAME):

        if db == 'NAConf':
            self.data_dir = 'MeynielMazancieux_NACONF_prob_2021'
        elif db == 'EncodeProb':
            self.data_dir = 'EncodeProb_BounmyMeyniel_2020'
        elif db == 'Explore':
            self.data_dir = 'Explore_Meyniel_Paunov_2021'
        elif db == 'PNAS':
            self.data_dir = 'Meyniel_MarkovGuess_2014'

        self.home_dir = '/home_local/alice_hodapp/NeuroModAssay'
        self.root_dir = '/neurospin/unicog/protocols/IRMf'
        self.receptor_path = '/home/ah278717/hansen_receptors/data/PET_nifti_images/' #path to downloaded data from Hansen et al. (2022)
        self.alpha_path = '/home/ah278717/alpha2_receptor/' #path to the data shared by Benedicte Ballanger

class Receptors:
    def __init__(self, source=RECEPTOR_SOURCE):
        self.source = RECEPTOR_SOURCE

        if source in ['PET', 'PET2']:
            self.receptor_names = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                                "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                                "MOR", "NET", "NMDA", "VAChT", "A2"]
            self.serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
            self.acetylcholine = ["A4B2", "M1", "VAChT"]
            self.noradrenaline = ["NET", "A2"]
            self.opioid = ["MOR"]
            self.glutamate = ["mGluR5", 'NMDA']
            self.histamine = ["H3"]
            self.gaba = ["GABAa"]
            self.dopamine = ["D1", "D2", "DAT"]
            self.cannabinnoid = ["CB1"]
            self.exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
            self.inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR', 'A2']

        if source == 'autorad_zilles44':
            self.receptor_names = ['AMPA', 'NMDA', 'kainate', 'GABAa', 'GABAa-BZ', 'GABAb', 'm1', 'm2', 'm3', 'a4b2',
                                'a1', 'a2', '5-HT1a', '5-HT2', 'D1'] 
            self.serotonin = ['5-HT1a', '5-HT2']
            self.acetylcholine = ['m1', 'm2', 'm3', 'a4b2']
            self.noradrenaline = ['a1', 'a2']
            self.glutamate = ['AMPA', 'NMDA', 'kainate']
            self.gaba = ['GABAa', 'GABAa-BZ', 'GABAb']
            self.dopamine = ['D1']
            self.exc = ['AMPA', 'NMDA', 'kainate', 'm1', 'm3', 'a4b2', 'a1', '5-HT2', 'D1']
            self.inh = ['GABAa', 'GABAa-BZ', 'GABAb', 'm2', 'a2', '5-HT1a']

        if source == 'AHBA':
            self.receptor_names = ['ADRA1A', 'ADRA1B', 'ADRA1D', 'ADRA2A', 'ADRA2C', 'ADRB1', 'ADRB2',
                                   'HTR1A', 'HTR1E', 'HTR2A', 'HTR3', 'HTR4','HTR7',
                                   'CHRM1', 'CHRM2', 'CHRM4', 'CHRNB2',
                                   'DRD1', 'DRD2', 'DRD4']
            self.noradrenaline = ['ADRA1A', 'ADRA1B', 'ADRA1D', 'ADRA2A', 'ADRA2C', 'ADRB1', 'ADRB2']
            self.serotonin = ['HTR1A', 'HTR1E', 'HTR2A', 'HTR3', 'HTR4','HTR7'] 
            self.acetylcholine = ['CHRM1', 'CHRM2', 'CHRM4', 'CHRNB2']
            self.dopamine = ['DRD1', 'DRD2', 'DRD4']
            self.exc = ['ADRA1A', 'ADRA1B', 'ADRA1D','ADRB1', 'ADRB2','HTR2A','HTR3','HTR4','HTR7','CHRM1','CHRNB2','DRD1']
            self.inh = ['ADRA2A', 'ADRA2C','HTR1A', 'HTR1E','CHRM2', 'CHRM4','DRD2', 'DRD4']


