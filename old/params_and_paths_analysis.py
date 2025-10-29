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
RECEPTOR_SOURCE = 'PET2' #PET2 is the dataset including alpha2
#---------------------------------------------------

class Params:
    def __init__(self):
        
        self.tasks = ['lanA'] #['EncodeProb', 'NAConf', 'PNAS', 'Explore'] # or ['lanA']
        self.latent_vars = ['S-N']#['confidence', 'surprise',] # or ['S-N'] for language

        self.ignore = {'NAConf': [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59], #30, 43, 15, 40, 42, 59 are removed because of their low coverage 
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': [],
            'lanA' : [80]} # random selection of subjects
        
        self.mask = 'schaefer'

        self.zscore_per_session = True 

        self.study_mapping = {
            "EncodeProb": "Study 1",
            "NAConf": "Study 2",
            "PNAS": "Study 3",
            "Explore": "Study 4",
        }


class Paths:
    def __init__(self):

        self.fmri_dir = {'NAConf': 'MeynielMazancieux_NACONF_prob_2021/derivatives',
                    'EncodeProb': 'EncodeProb_BounmyMeyniel_2020/derivatives',
                    'Explore': 'Explore_Meyniel_Paunov_2021/bids/derivatives/fmriprep-23.1.3_MAIN',
                    'PNAS': 'Meyniel_MarkovGuess_2014/MRI_data/analyzed_data',
                    'lanA': '/home_local/alice_hodapp/language_localizer'}

        self.home_dir = '/home_local/alice_hodapp/NeuroModAssay'
        self.root_dir = '/neurospin/unicog/protocols/IRMf'

        self.receptor_path = '/home/ah278717/hansen_receptors/data/PET_nifti_images/' #path to downloaded data from Hansen et al. (2022)
        self.alpha_path = '/home/ah278717/alpha2_receptor/' #path to the data shared by Benedicte Ballanger

class Receptors:
    def __init__(self, source=RECEPTOR_SOURCE):
        self.source = RECEPTOR_SOURCE

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

        if source == 'PET2':
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

        # Grouping 
        self.receptor_groups = [self.serotonin, self.acetylcholine, self.noradrenaline, self.opioid, self.glutamate, self.histamine, self.gaba, self.dopamine, self.cannabinnoid]
        self.group_names = ['serotonin', 'acetylcholine', 'norepinephrine', 'opioid', 'glutamate', 'histamine', 'gaba', 'cannabinnoid']
        self.receptor_class = [self.exc,self.inh]

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


