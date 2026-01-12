#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
MASK_NAME =  'schaefer' 
PARCELATED = False
PARCELATED = False

#exploratory setting
UPDATE_REG = False #update or suprise + confidence as regressor
#---------------------------------------------------

class Params:
    def __init__(self, task, cv_true=False, mask=MASK_NAME, parcel = PARCELATED, update=UPDATE_REG):

        self.tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
         # random selection of subjects
        if task == "language":
            self.tasks = ['lanA'] 
            self.ignore =  [80] # random selection of subjects
        elif task == 'EncodeProb':
            self.db = task
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = 5 #no smoothing necessary because of 7T fMRI -> just for comparison with other studies 
            self.ignore = [1, 4, 12, 20]
            self.session = {6:[1,3,4,5], 20:[1,2,3,5], 21:[1,2,3,5]}
        elif task == 'NAConf':
            self.db = task
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = 5
            self.naconf_behav_subj = \
            [1, 2, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
            43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 61] 
            self.ignore = [3, 5, 6, 9, 36, 51] if cv_true == False else [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59] #30, 43, 15, 40, 42, 59 are removed for CV only because of their low coverage 
            self.session = []
        elif task == 'PNAS':
            self.db = task
            self.seq_type = 'transition'
            self.smoothing_fwhm = 5
            self.ignore = []
            self.session = []
        elif task == 'Explore':
            self.db = task
            self.smoothing_fwhm = 5
            self.ignore = [9, 17, 46]
            self.session = []
            self.split = False 
            self.reward = False 
            self.model = 'noEntropy_noER'  #[noEntropy]
            self.io_variables = ['US', 'EC_chosen'] #['US', 'EC_chosen, 'ER_chosen']

        if task != 'Explore':
            self.io_options = {'p_c': 1/75, 'resol': 20} 

        if task == 'language':
            self.latent_vars = ['S-N']
        else:
            self.latent_vars = ['confidence', 'surprise']
            self.variables_long = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

        self.mask = mask

        self.parcelated = parcel
        self.update = update

        self.hpf = 1/128 
        self.hrf = 'spm'
        
        self.mask = mask

        self.zscore_per_session = True 

        self.study_mapping = {
            "EncodeProb": "Study 1",
            "NAConf": "Study 2",
            "PNAS": "Study 3",
            "Explore": "Study 4",
        }

        #mask details
        if mask == 'schaefer':
            self.mask_details = '100' #number of regions
        elif mask == 'desikan':
            self.mask_details = '68'

class Paths:
    def __init__(self, task):

        if task == 'NAConf':
            self.data_dir = 'MeynielMazancieux_NACONF_prob_2021'
            self.mov_dir = 'derivatives'
        elif task == 'EncodeProb':
            self.data_dir = 'EncodeProb_BounmyMeyniel_2020'
            self.mov_dir = 'derivatives'
        elif task == 'Explore':
            self.data_dir = 'Explore_Meyniel_Paunov_2021'
            self.mov_dir = 'bids/derivatives/fmriprep-23.1.3_MAIN'
        elif task == 'PNAS':
            self.data_dir = 'Meyniel_MarkovGuess_2014'
            self.mov_dir = 'MRI_data/raw_data'

        self.home_dir = '/home_local/alice_hodapp/NeuroModAssay'
        self.root_dir = '/neurospin/unicog/protocols/IRMf'

        self.receptor_path = '/home/ah278717/hansen_receptors/data/PET_nifti_images/' #path to downloaded data from Hansen et al. (2022)
        self.alpha_path = '/home/ah278717/alpha2_receptor/' #path to the data shared by Benedicte Ballanger

class Receptors:
    def __init__(self, source):

        self.source = source

        if source == 'PET':
            self.receptor_names = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                                "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                                "MOR", "NET", "NMDA", "VAChT"]
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
            self.inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']

            self.receptor_label_formatted = [
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$', '$5\\text{-}\\mathrm{HTT}$',
                '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$', '$\\mathrm{M}_{\\mathrm{1}}$',
                '$\\mathrm{VAChT}$', '$\\mathrm{NET}$', 
                '$\\mathrm{MOR}$', '$\\mathrm{mGluR}_{\\mathrm{5}}$', '$\\mathrm{NMDA}$',
                '$\\mathrm{H}_{\\mathrm{3}}$', '$\\mathrm{GABA}_{\\mathrm{a}}$', '$\\mathrm{D}_{\\mathrm{1}}$',
                '$\\mathrm{D}_{\\mathrm{2}}$', '$\\mathrm{DAT}$', '$\\mathrm{CB}_{\\mathrm{1}}$'
            ]   

        #PET2 is the dataset including alpha2
        elif source == 'PET2':
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

            self.receptor_label_formatted = [
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$', '$5\\text{-}\\mathrm{HTT}$',
                '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$', '$\\mathrm{M}_{\\mathrm{1}}$',
                '$\\mathrm{VAChT}$', '$\\mathrm{NET}$', '$\\mathrm{A}_{\\mathrm{2}}$',
                '$\\mathrm{MOR}$', '$\\mathrm{mGluR}_{\\mathrm{5}}$', '$\\mathrm{NMDA}$',
                '$\\mathrm{H}_{\\mathrm{3}}$', '$\\mathrm{GABA}_{\\mathrm{a}}$', '$\\mathrm{D}_{\\mathrm{1}}$',
                '$\\mathrm{D}_{\\mathrm{2}}$', '$\\mathrm{DAT}$', '$\\mathrm{CB}_{\\mathrm{1}}$'
            ]

        # Grouping 
        self.receptor_groups = [self.serotonin, self.acetylcholine, self.noradrenaline, self.opioid, self.glutamate, self.histamine, self.gaba, self.dopamine, self.cannabinnoid]
        self.group_names = ['serotonin', 'acetylcholine', 'norepinephrine', 'opioid', 'glutamate', 'histamine', 'gaba', 'cannabinnoid']
        self.receptor_class = [self.exc,self.inh]



