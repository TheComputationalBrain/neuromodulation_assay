#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:59:22 2024

@author: Alice Hodapp
"""

#----------------------------------------------------
#  PARAMS TO CHANGE   #
DB_NAME = 'Explore' # other options: 'EncodeProb','NAConf', 'Explore', 'PNAS'
MASK_NAME = 'harvard_oxford_cortical' #'harvard_oxford_cortical'; harvard_oxford_subcortical; schaefer, desikan
PARCELATED = False
RECEPTOR_SOURCE = 'PET2' #,'PET', '$PET' or 'autorad_zilles44', 'AHBA', #PET2 is the dataset including alpha2

#fixed at best setting:
UPDATE_REG = False #update or suprise + confidence as regressor
#---------------------------------------------------

class Params:
    def __init__(self, db=DB_NAME, mask = MASK_NAME, parcel = PARCELATED, update =UPDATE_REG):
        self.db = db
        self.mask = mask

        self.parcelated = parcel
        self.update = update

        self.hpf = 1/128
        self.hrf = 'spm'

        if db != 'Explore':
            self.io_options = {'p_c': 1/75, 'resol': 20} 

        if db == 'EncodeProb':
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = None #no smoothing necessary because of 7T fMRI
        elif db == 'NAConf':
            self.seq_type = 'bernoulli'
            self.smoothing_fwhm = 2
        elif db == 'PNAS':
            self.seq_type = 'transition'
            self.smoothing_fwhm = 5
        elif db == 'Explore':
            self.smoothing_fwhm = 5
            self.split = True #Split free and forced trials 


        if UPDATE_REG:
            self.latent_vars = ['update', 'predictability', 'predictions']
        else:
            self.latent_vars = ['surprise', 'confidence', 'predictability', 'predictions']

        #participants to ignore and session deviations
        if db == 'NAConf':
            self.ignore = [3, 5, 6, 9, 36, 51, 54]
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
            self.subnums_explore = {}
            self.io_variables = ['EU_chosen_z', 'US_z', 'ER_chosen_z', 'entropy_chosen_z'] #TODO: try chosen-unchosen?
            self.latent_vars = ['surprise_free', 'confidence_free', 'predictability_free', 'predictions_free',
                                'surprise_forced', 'confidence_forced', 'predictability_forced', 'predictions_forced']
        
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


class Receptors:
    def __init__(self, source=RECEPTOR_SOURCE):
        self.source = RECEPTOR_SOURCE

        if source in ['PET', 'PET2']:
            self.receptor_names = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                                "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                                "MOR", "NET", "NMDA", "VAChT", "a2"]
            self.serotonin = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"]
            self.acetylcholine = ["A4B2", "M1", "VAChT"]
            self.noradrenaline = ["NET", "a2"]
            self.opioid = ["MOR"]
            self.glutamate = ["mGluR5", 'NMDA']
            self.histamine = ["H3"]
            self.gaba = ["GABAa"]
            self.dopamine = ["D1", "D2", "DAT"]
            self.cannabinnoid = ["CB1"]
            self.exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
            self.inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']

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


