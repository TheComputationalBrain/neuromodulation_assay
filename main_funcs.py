#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:49:30 2024

@author: Alice hodapp, Maëva L'Hôtellier
"""

import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import os.path as op
from scipy.io import loadmat
from initialize_subject import initialize_subject
from params_and_paths import Params, Paths

params = Params()
paths = Paths()

def demean(x):
    """center regressors"""
    mean = np.nanmean(x)
    return x - mean

def get_json_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):
   
    json_files_dir = {'NAConf': op.join(root_dir, data_dir, 'bids_dataset'),
                    'EncodeProb': op.join(root_dir, data_dir, 'bids_dataset'),
                    'Explore': op.join(root_dir, data_dir, 'bids/raw/'),
                    'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}
    
    return json_files_dir[db_name]


def get_fmri_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):
    
    fmri_dir = {'NAConf': op.join(root_dir, data_dir, 'derivatives'),
                'EncodeProb': op.join(root_dir, data_dir, 'derivatives'),
                'Explore': op.join(root_dir, data_dir, 'bids/derivatives/fmriprep-23.1.3_MAIN'),
                'PNAS': op.join(root_dir, data_dir, 'MRI_data/analyzed_data')}

    return fmri_dir[db_name]


def get_beh_dir(db_name, root_dir=paths.root_dir, data_dir=paths.data_dir):

    beh_dir = {'NAConf': op.join(root_dir, data_dir),
               'EncodeProb': op.join(root_dir, data_dir),
               'Explore': '/home_local/EXPLORE/github/explore/2021_Continous_2armed',
               'PNAS': op.join(root_dir, data_dir)}

    return beh_dir[db_name]

def get_subjects(db, data_dir): 
    subjects = []

    if db == 'lanA':
        subjects = list(range(1, 61))
        return subjects

    if db != 'PNAS':
        folders = os.path.join(data_dir,
                            'sub-*')
        
        for folder in glob.glob(folders):
            if os.path.isdir(folder):  # Ensure the path is a directory
                # Extract the number from the filename
                number = folder.split('sub-')[1]
                subjects.append(int(number))
    else:
        folders = os.path.join(data_dir,
                            'subj*')
        
        for folder in glob.glob(folders):
            # Extract the number from the filename
            number = folder.split('subj')[1]
            subjects.append(int(number))

    #keep this as a test for now 
    nsub_correct = {'PNAS': 21,
            'EncodeProb': 30,
            'NAConf': 56,
            'Explore': 59}
    
    if len(subjects) != nsub_correct[db]:
        raise ValueError(f"Found {len(subjects)} subjects but experiment has: {nsub_correct[db]}.")

    return sorted(subjects)

def convert_to_secs(data, var):
    return (data[var].dropna().values - data['t_trigger'][0])/1000

def get_seq(db, sub, sess, beh_dir):

    if db == 'Explore':

        file = glob.glob(os.path.join(beh_dir, 'Data_neurospin/formatted/fmri_session',
                            f'sub-{sub:02d}/data_subject_*_session_{sess}.csv'))[0]
        seq = pd.read_csv(file, index_col=0)

    else:
          
        if db == 'EncodeProb':
            filepath = glob.glob(os.path.join(beh_dir,
                                f'behavior/sub-{sub:02d}*',
                                f'*sess_{sess}.csv'))[0]
    
            file = pd.read_csv(filepath, header=1)
            seq  = file['s'].values

        if db == 'NAConf':
            file = op.join(beh_dir,
                                'behaviour_eyeTracker_data/predefined_sequences',
                                f'sequence_{sess-1}.pickle')
            with open(file, 'rb') as f:
                predefined = pickle.load(f)
                seq = predefined['sequence']+1

        if db == 'PNAS':
            beh_data = get_data_PNAS(sub, sess, beh_dir)
            seq= beh_data['sequence']
    
        seq = np.where(seq==1, 0, np.where(seq==2, 1, seq))[~np.isnan(seq)]

    return seq

def get_data_PNAS(sub, sess, data_dir):
    
    filespath = os.path.join(data_dir,'Behavioral_data')
    beh_file = glob.glob(os.path.join(filespath,
                                      f'*SUBJECT_{sub}_Sess_{sess+1}_*'))
    if len(beh_file) > 1:
        raise('check behavioral files PNAS')

    beh_data = loadmat(beh_file[0])
    T0 = beh_data['save_T0'][0]

    indices_question = beh_data['StimQ'][0].astype(int)-1
    task_data = {
        'sequence': beh_data['s'][0],
        'stim_onset': beh_data['save_tOnStimV'][:, 0] - T0,
        'question_onset': beh_data['save_tOnConf'][:, 0][indices_question] - T0,
        'response_onset': beh_data['save_tConfValid'][:, 0][indices_question] - T0,
        'subj_confidence': beh_data['save_ConfRating'][:, 0][indices_question],
        'modality': beh_data['Modality'][0]}
    task_data['reaction_times'] = task_data['response_onset'] - task_data['question_onset']
    question_lenght = [int(beh_data['t_BeforePredictionQ'])]
    task_data['question_lenght'] = question_lenght * len(task_data['question_onset'])

    return task_data


def import_json_info(data_dir, sub, sess):
    """This function imports the information from the .json file containing
    metadata about a run acquisition."""

    json_file = glob.glob(os.path.join(data_dir,
                                  f'sub-{sub:02d}/func',
                                  f'*run-{sess:02d}*.json'))
    if not json_file:
        raise FileNotFoundError('JSON file not found.')
    return json.load(open(json_file[0]))


def rescale_answer(pos, pos_min, pos_max):
    """This function rescales an answer onto a new scale that features pos_min
    and pos_max as extreme values.
    """
    answer = (pos + pos_max) / (pos_max - pos_min)
    return answer

def get_events_explore(sub, sess, nan_missed=True):

    #confidence = 1-EU (estimation confidence is the same); 
    #surprise = surprise at outcome/shannon surprise
    #predictabiliy = compute from prior that is returned from IO observer 
    #prediction = expected reward
    #(at outcome = prediction error and surprise of previous trial)
    
    # events_fixed = ['cue','out', 'qA_val', 'qA_conf', 'qB_val', 'qB_conf','respA', 'respB'] #will be added as an unmodulated regressor
    # if params.reward:
    #     events_modulated = ['rt_respA', 'rt_respB', 'ER_diff', 'reward','sub_qA_val', 'io_qA_val', 'rt_qA_val', 'sub_qA_conf', 'io_qA_conf', 'rt_qA_conf',
    #                     'sub_qB_val', 'io_qB_val', 'rt_qB_val', 'sub_qB_conf','io_qB_conf', 'rt_qB_conf']
    # else:
    #     events_modulated = ['rt_respA', 'rt_respB', 'ER_diff', 'sub_qA_val', 'io_qA_val', 'rt_qA_val', 'sub_qA_conf', 'io_qA_conf', 'rt_qA_conf',
    #                     'sub_qB_val', 'io_qB_val', 'rt_qB_val', 'sub_qB_conf','io_qB_conf', 'rt_qB_conf']

    events_fixed = ['cue','out', 'qA_val', 'qA_conf', 'qB_val', 'qB_conf']
    events_modulated = ['ER_diff', 'sub_qA_val', 'io_qA_val', 'rt_qA_val', 'sub_qA_conf', 'io_qA_conf', 'rt_qA_conf',
                        'sub_qB_val', 'io_qB_val', 'rt_qB_val', 'sub_qB_conf','io_qB_conf', 'rt_qB_conf']
    event_labels = events_fixed + events_modulated + params.io_variables

    event_label_split = []

    if params.split:
        for event in event_labels:
            if 'q' in event:
                event_label_split.append(f"{event}_all") #questions and answers stay combined
            else:
                event_label_split.append(f"{event}_free") #split other modulators 
                event_label_split.append(f"{event}_forced")

    io_times = []
    for reg in params.io_variables:
        io_times.append(f'{reg}_start')

    # if params.reward:
    #     time_cols = ['trial_start', 'outcome_start', 'qA_val_start', 'qA_conf_start', 'qB_val_start', 'qB_conf_start', 'startA', 'startB', 
    #                     'rt_startA', 'rt_startB', 'ER_diff_start', 'reward_start',
    #                     'qA_val_sub_start', 'qA_val_io_start', 'qA_val_rt_start', 'qA_conf_sub_start', 'qA_conf_io_start', 'qA_conf_rt_start',
    #                     'qB_val_sub_start', 'qB_val_io_start', 'qB_val_rt_start', 'qB_conf_sub_start', 'qB_conf_io_start', 'qB_conf_rt_start'
    #                     ]  + io_times
    # else:
    #     time_cols = ['trial_start', 'outcome_start', 'qA_val_start', 'qA_conf_start', 'qB_val_start', 'qB_conf_start', 'startA', 'startB', 
    #                     'rt_startA', 'rt_startB','ER_diff_start', 
    #                     'qA_val_sub_start', 'qA_val_io_start', 'qA_val_rt_start', 'qA_conf_sub_start', 'qA_conf_io_start', 'qA_conf_rt_start',
    #                     'qB_val_sub_start', 'qB_val_io_start', 'qB_val_rt_start', 'qB_conf_sub_start', 'qB_conf_io_start', 'qB_conf_rt_start'
    #                     ]  + io_times

    time_cols = ['trial_start', 'outcome_start', 'qA_val_start', 'qA_conf_start', 'qB_val_start', 'qB_conf_start', 'ER_diff_start', 
                            'qA_val_sub_start', 'qA_val_io_start', 'qA_val_rt_start', 'qA_conf_sub_start', 'qA_conf_io_start', 'qA_conf_rt_start',
                            'qB_val_sub_start', 'qB_val_io_start', 'qB_val_rt_start', 'qB_conf_sub_start', 'qB_conf_io_start', 'qB_conf_rt_start'
                            ]  + io_times

    arm_ids = ['A', 'B']

    para = initialize_subject(sub)# no need to specify vol because we only get onsets from this function
    #para = para.set_index(np.arange(0, para.shape[0]))

    #by block
    para = para[para['block'] == sess].reset_index(drop=True)
    #ignore first 4 or 8  trials as activity + confidence increases dramatically (but unrelated)
    if params.remove_trials: #remove the first 4 trials -> does this reduce the positive component?
        para = para.iloc[8:].reset_index(drop=True)
    else:
        para = para.iloc[4:].reset_index(drop=True)
        
    n_trials = len(para) #get trial n from length of df

    if params.split:
        trial_types_all = np.asarray(event_label_split * n_trials)
    else:
        trial_types_all = np.asarray(event_labels * n_trials)
    #trial_types_all = [trial_name.replace('EU', '1-EU') for trial_name in trial_types_all] #change name to represent the convertion to confidence
    #trial_types_all = np.asarray(trial_types_all)

    # Get onsets for missed trials #din't add a missed regressor since missed trials only appear in a few trial and prevent the creation of a design matrix by block
    # para['missed_start'] = np.nan
    # para.loc[para['ismissed'] == 1, 'missed_start'] = \
    #     para['trial_start'][para['ismissed'] == 1]

    # Don't count missed trials as regular trials
    if nan_missed:
        para.loc[para['ismissed'] == 1, 'trial_start'] = np.nan
        para.loc[para['ismissed'] == 1, 'outcome_start'] = np.nan
    para['rt_start'] = np.nansum(para[['trial_start', 'rt']], axis=1)
    para.loc[para['rt_start'] == 0, 'rt_start'] = np.nan

    # add additional time columns 
    para['startA'] = para['rt_start'].where(para['choiceA'] == 1, None)
    para['startB'] = para['rt_start'].where(para['choiceA'] == 0, None)
    para['rt_startA'] = para['startA']
    para['rt_startB'] = para['startB']
    para['ER_diff_start'] = para['trial_start']
    if params.reward:
        para['reward_start'] = para['outcome_start'] 

    # ADD QUESTION PARAMTERS
    #infer onset of questions
    ind_Q1_is_A = para.index[para['rtQ1_val'] == para['rtA_val']]
    ind_Q1_is_B = para.index[para['rtQ1_val'] == para['rtB_val']]

    for var in ['qA_val_start', 'qA_conf_start', 
                'qB_val_start', 'qB_conf_start']:
        para[var] = np.nan

    #I changed this to match the encodeProb questions: onset of question with duration of RT and modulation of subj response
    for i in ind_Q1_is_A: 
        para['qA_val_start'][i] =  para['trial_end'][i] + 1
        para['qA_conf_start'][i] = para['qA_val_start'][i] + para['rtA_val'][i] + 0.5  
        para['qB_val_start'][i] = para['qA_conf_start'][i] + 1 
        para['qB_conf_start'][i] = para['qB_val_start'][i] + para['rtB_val'][i] + 1 

    for i in ind_Q1_is_B: 
        para['qB_val_start'][i] = para['trial_end'][i] + 1
        para['qB_conf_start'][i] = para['qB_val_start'][i] + para['rtB_val'][i] + 0.5
        para['qA_val_start'][i] = para['qB_conf_start'][i] + 1
        para['qA_conf_start'][i] = para['qA_val_start'][i] + para['rtA_val'][i] + 1 

    # #duplicate for sub and IO estimates + RT regressor #!remove the missing and redundand trials here!!
    # for arm in arm_ids:
    #     for l in ['val', 'conf']:
    #         mask = para[f'q{arm}_{l}_start'].notna() & para[f'opt{arm}_{l}'].isna()
    #         for e in ['sub', 'io', 'rt']:
    #             para.loc[mask, f'q{arm}_{l}_{e}_start'] = np.nan  # Set NaN where no answer given
    #             para.loc[~mask, f'q{arm}_{l}_{e}_start'] = para.loc[~mask, f'q{arm}_{l}_start']  # Otherwise, assign original value

        # #duplicate for sub and IO estimates + RT regressor #!remove the missing and redundand trials here!!
    for arm in arm_ids:
        for l in ['val', 'conf']:
            for e in ['sub', 'io', 'rt']:
                para[f'q{arm}_{l}_{e}_start'] = para[f'q{arm}_{l}_start']  

    #add columns for durations
    for dur in events_fixed:
        para[f'{dur}_drt'] = 0
    for arm in arm_ids:
        for l in ['val', 'conf']:
            para[f'q{arm}_{l}_drt'] = para[f'rt{arm}_{l}']    
    # para['rt_startA_drt'] = 0 
    # para['rt_startB_drt'] = 0
    para['ER_diff_drt'] = 0
    if params.reward:
        para['reward_drt'] = 0
    for arm in arm_ids:
        for l in ['val', 'conf']:
            for e in ['sub', 'io', 'rt']:
                if e != 'rt':
                    para[f'q{arm}_{l}_{e}_drt'] = para[f'rt{arm}_{l}'] #duration of question regressors = RTs 
                else:
                    para[f'q{arm}_{l}_{e}_drt'] = 0

    #add columns for modulation  
    for mod in events_fixed:
        para[f'{mod}_mod'] = 1
    # para['rt_startA_mod'] = para['rt'].where(para['choiceA'] == 1, None)
    # para['rt_startB_mod'] = para['rt'].where(para['choiceA'] == 0, None)
    para['ER_diff_mod'] = para['ER_diff']
    if params.reward:
        para['reward_mod'] = para['reward'] 
    for arm in arm_ids:
        for l in ['val', 'conf']:
            para[f'q{arm}_{l}_sub_mod'] = para[f'opt{arm}_{l}']
            para[f'q{arm}_{l}_sub_mod'] = para[f'opt{arm}_{l}'].fillna(0) #! for now just replace it with 0 
            io_est = np.full(len(para), np.nan)
            if l == 'val':
                io_est[~para[f'q{arm}_val_start'].isna()] = para.loc[~para[f'q{arm}_val_start'].isna(), f'ER_{arm}']
            else:
                io_est[~para[f'q{arm}_conf_start'].isna()] = para.loc[~para[f'q{arm}_conf_start'].isna(), f'EC_{arm}']
            para[f'q{arm}_{l}_io_mod'] = io_est
            para[f'q{arm}_{l}_rt_mod'] = para[f'rt{arm}_{l}']

    # ADD IO RESULTS OF INTEREST
    #currently all estimates are releated to the chosen option
    #other option that produces similar maps as EncodeProb is chosen-unchosen
    for reg in params.io_variables:
        para[f'{reg}_start'] = para['outcome_start'] # lock IO regressors to outcome to stay as close as possible to the other datasets
        para[f'{reg}_drt'] = 0
        para[f'{reg}_mod'] = para[reg]

    if params.split:
        # SPLIT FREE AND FORCED 
        dur_cols = [col for col in para.columns if '_drt' in col]
        mod_cols = [col for col in para.columns if '_mod' in col]
        all_cols = time_cols + dur_cols + mod_cols 

        for col in all_cols:
            # Get the value of 'isfree'
            if 'q' in col:
                para[f'{col}_all'] = para[col] #questions and answers stay combined
            else:
                isfree = para['isfree']

                # Get the original value of the column
                original_array = para[col]

                free_array = []
                forced_array = []

                for i, value in enumerate(original_array):
                    if isfree[i] == 1:
                        free_array.append(value)
                        forced_array.append(float('nan'))  
                    else:
                        free_array.append(float('nan')) 
                        forced_array.append(value)
                
                # Assign the new free and forced arrays to the para dictionary
                para[f'{col}_free'] = free_array
                para[f'{col}_forced'] = forced_array
        time_cols = [col for col in para if '_start' in col and ('_free' in col or '_forced' in col or '_all' in col)] #only the seperated predictors 
        dur_cols = [col for col in para if '_drt' in col and ('_free' in col or '_forced' in col or '_all' in col)]
        mod_cols = [col for col in para if '_mod' in col and ('_free' in col or '_forced' in col or '_all' in col)]
    else:
        mod_cols = [col for col in para if '_mod' in col]
        dur_cols = [col for col in para if '_drt' in col]

    onsets = []
    onsets.append(para[time_cols].values)
    onsets = np.hstack(onsets).ravel('C')
    trial_types = trial_types_all[~np.isnan(onsets)]
    durations = []
    durations.append(para[dur_cols].values)
    durations = np.hstack(durations).ravel('C')
    durations = durations[~np.isnan(onsets)] 
    modulations = []
    for col in mod_cols: #deman the free and forced trials seperatly
        if not para[col].dropna().eq(1).all(): #skip the collumns that contain only a constant
            para[col] = demean(para[col])
    modulations.append(para[mod_cols].values)
    modulations = np.hstack(modulations).ravel('C')
    modulations = modulations[~np.isnan(onsets)]
    onsets = onsets[~np.isnan(onsets)]

    events = pd.DataFrame({'onset': onsets,
                            'trial_type': trial_types,
                            'duration': durations,
                        'modulation': modulations})

    filtered_events = events.copy()

    # for arm in arm_ids:
    #     for l in ['val', 'conf']:
    #         trial_type_sub = f"sub_q{arm}_{l}"
    #         trial_type_io = f"io_q{arm}_{l}"

    #         # Check if modulation is consistent for sub_q{arm}_{l}
    #         if filtered_events.loc[filtered_events['trial_type'] == trial_type_sub, 'modulation'].nunique() == 1:
    #             # Remove both sub_q{arm}_{l} and io_q{arm}_{l}
    #             filtered_events = filtered_events[~filtered_events['trial_type'].isin([trial_type_sub, trial_type_io])]

    return events

def get_sessions(sub):
    if sub in params.session:
        return params.session[sub]
    else:
        return [1,2,3,4]
    
def get_stimq(db, sub, sess, data_dir):
    if db == 'EncodeProb':
        filepath = glob.glob(op.join(data_dir,
                                    'behavior',
                                    f'sub-{sub:02d}*',
                                    f'*session_{sess}.pickle'))[0]
        data = pickle.load(open(filepath, 'rb'))
        indices_question = data['StimQ'][0].astype(int)
    elif db == 'PNAS':
        filespath = os.path.join(data_dir,'Behavioral_data')
        beh_file = glob.glob(os.path.join(filespath,
                                        f'*SUBJECT_{sub}_Sess_{sess+1}_*'))
        beh_data = loadmat(beh_file[0])
        indices_question = beh_data['StimQ'][0].astype(int)-1
    elif db == 'NAConf':
        behav_dir = os.path.join(data_dir, 'behaviour_data', f'sub-{sub:02d}')
        info = glob.glob(os.path.join(behav_dir, f'experiment_info_sub-{sub:02d}.pickle'))[0]
        with open(info, 'rb') as f:
            exp = pickle.load(f)
        indices_question = [int(stim) for stim in exp[sess-1]['stim_q']]


    return indices_question


def get_events(db, sub, sess, data_dir=None, io_inference=None, seq=None): 

    if db == 'Explore':
        events = get_events_explore(sub, sess)
        return events
    else:

        #add IO regressors
        if params.update:
        # IO regressors
            io_regs = pd.DataFrame({'update': io_inference['update'],
                                    'predictions': io_inference['p1'],
                                    'predictability': io_inference['entropy']})
        else:
            io_regs = pd.DataFrame({'surprise': io_inference['surprise'],
                                    'confidence': io_inference['confidence_pre'],
                                    'predictions': io_inference['p1'],
                                    'predictability': io_inference['entropy']})

        stim_q = get_stimq(db, sub, sess, data_dir)
        if stim_q[-1] >= len(seq): 
            stim_q = stim_q[:-1]

        io_conf_q = np.array([-np.log(io_inference['p1_sd_array'])[q]
                for q in stim_q])
    
        if db == 'PNAS': 
            data = get_data_PNAS(sub, sess, data_dir)
            onsets = np.hstack((data['stim_onset'], data['question_onset'], data['question_onset'],data['question_onset'],data['question_onset']))
            duration = np.hstack(([0] * len(data['stim_onset']),
                                data['reaction_times'], data['reaction_times'], data['reaction_times'],[0] * data['question_onset']))
            trial_type = np.hstack((['stim']*len(data['stim_onset']),
                                    ['q_conf']*len(data['question_onset']),
                                    ['sub_conf']*len(data['question_onset']),
                                    ['io_conf']*len(data['question_onset']),
                                    ['rt_conf']*len(data['question_onset'])))
            modulation = np.hstack(([1] * len(data['stim_onset']),
                                    [1] * len(data['question_onset']), 
                                    demean(data['subj_confidence']),
                                    demean(io_conf_q),
                                    demean(data['reaction_times']))) 
            on_stim = data['stim_onset']

        elif db == 'EncodeProb':
            filepath = glob.glob(os.path.join(data_dir,
                            'behavior',
                            f'sub-{sub:02d}*',
                            f'*sess_{sess}.csv'))[0]
            data = pd.read_csv(filepath, header=1)
            on_stim = convert_to_secs(data, 't_on_stim')

            on_q_prob = convert_to_secs(data,'t_question_prob_on')
            on_q_conf = convert_to_secs(data,'t_question_conf_on')
            rt_prob = data['estim_rt'].dropna().values/1000
            rt_conf = data['conf_rt'].dropna().values/1000

            sub_prob = convert_to_secs(data, 'estim_position')
            sub_prob = rescale_answer(sub_prob, pos_min=-500,pos_max=500)
            sub_conf = convert_to_secs(data, 'conf_position')
            sub_conf = rescale_answer(sub_prob, pos_min=-500,pos_max=500)

            io_prob_q = np.array([io_inference['p1_mean_array'][q]
                        for q in stim_q])

            onsets = np.hstack((on_stim, on_q_prob, on_q_prob, on_q_prob, on_q_prob, on_q_conf, on_q_conf, on_q_conf, on_q_conf))
            trial_type = np.hstack((['stim'] * len(on_stim),
                                    ['q_prob'] * len(on_q_prob),
                                    ['sub_prob'] * len(on_q_prob),
                                    ['io_prob'] * len(on_q_prob),
                                    ['rt_prob'] * len(on_q_prob),
                                    ['q_conf'] * len(on_q_conf),
                                    ['sub_conf'] * len(on_q_conf),
                                    ['io_conf'] * len(on_q_conf),
                                    ['rt_conf'] * len(on_q_conf)))
            duration = np.hstack(([0] * on_stim,
                                rt_prob, rt_prob, rt_prob, [0] * on_q_prob, 
                                rt_conf, rt_conf, rt_conf, [0] * on_q_conf))
            modulation = np.hstack(([1] * len(on_stim), 
                                    [1] * len(on_q_prob),
                                    demean(sub_prob), 
                                    demean(io_prob_q),
                                    demean(rt_prob), 
                                    [1] * on_q_conf,
                                    demean(sub_conf),
                                    demean(io_conf_q),
                                    demean(rt_prob)))
                    
        elif db == 'NAConf':
            filespath = os.path.join(data_dir,
                    'behaviour_data',
                    f'sub-{sub:02d}')
            
            # load the experiment info file present in all subject behavioral folder
            file = os.path.join(filespath, f'experiment_info_sub-{sub:02d}.pickle')
            with open(file, 'rb') as f:
                    exp = pickle.load(f)

            if params.remove_trials: #remove the first 8 trials -> does this reduce the positive component?
                exp[sess-1]['stim_onsets'] = exp[sess-1]['stim_onsets'][8:]
                io_regs = io_regs.iloc[8:].reset_index(drop=True)
                exp[sess-1]['durations'] = exp[sess-1]['durations'][8:]

            # create event dataframe
            on_stim = exp[sess-1]['stim_onsets']
            on_q_prob = exp[sess-1]['question_prob_onsets']
            on_q_conf = exp[sess-1]['question_conf_onsets']

            rt_prob = exp[sess-1]['rt_prob']
            rt_conf = exp[sess-1]['rt_conf']
            rt_conf_clean = np.nan_to_num(exp[sess-1]['rt_conf'], nan=4) 
            rt_prob_clean = np.nan_to_num(exp[sess-1]['rt_prob'], nan=4)

            #create a variable for the onset of question modulators that were not missed (sice we have no values for the missed ones)
            #there is a onset regressor for all questions
            not_missed = exp[sess-1]['is_not_missed']
            on_q_prob_responded = on_q_prob[not_missed]
            on_q_conf_responded = on_q_conf[not_missed]
            #resp = exp[sess-1]['response_onsets']

            onsets = np.hstack((on_stim,
                                on_q_prob,
                                on_q_prob_responded,
                                on_q_prob_responded,
                                on_q_prob_responded,
                                on_q_conf,
                                on_q_conf_responded,
                                on_q_conf_responded,
                                on_q_conf_responded))

            trial_type = np.hstack((['stim']* len(on_stim), 
                                    ['q_prob']* len(on_q_prob),
                                    ['sub_prob']* len(on_q_prob_responded),
                                    ['io_prob']* len(on_q_prob_responded),
                                    ['rt_prob']* len(on_q_prob_responded),
                                    ['q_conf']*len(on_q_conf),
                                    ['sub_conf']* len(on_q_conf_responded),
                                    ['io_conf']* len(on_q_conf_responded),
                                    ['rt_conf']* len(on_q_conf_responded)))
  
            duration = exp[sess-1]['durations'] #durations are always 0
 
            duration = np.hstack((duration, 
                                    rt_prob_clean, 
                                    rt_prob[not_missed],
                                    rt_prob[not_missed],
                                    [0] * on_q_prob_responded,
                                    rt_conf_clean,
                                    rt_conf[not_missed],
                                    rt_conf[not_missed],
                                    [0] * on_q_conf_responded)) 
            
            sub_prob = exp[sess-1]['sub_prob']
            sub_conf = exp[sess-1]['sub_conf']

            io_prob_q = np.array([io_inference['p1_mean_array'][q]
                        for q in stim_q])
                    
            modulation = np.hstack(([1] * len(on_stim), 
                        [1] * len(on_q_prob),
                        demean(sub_prob[not_missed]), 
                        demean(io_prob_q[not_missed]), 
                        demean(rt_prob[not_missed]),
                        [1] * len(on_q_conf),
                        demean(sub_conf[not_missed]),
                        demean(io_conf_q[not_missed]),
                        demean(rt_conf[not_missed])))
            
        #add IO regressors 
        for column in io_regs.columns:
            regs = io_regs[column]
            onsets = np.concatenate((onsets, on_stim))
            duration = np.concatenate((duration, [0] * len(on_stim)))
            trial_type = np.concatenate((trial_type, [column] * len(on_stim)))
            centered_values = demean(regs.to_numpy()) #center all io modulation values 
            modulation = np.concatenate((modulation, centered_values))

        events = pd.DataFrame({'onset': onsets,
                            'duration': duration,
                            'trial_type': trial_type,
                            'modulation': modulation
                            })            

    return events

def convert_to_secs(data, var):
    return (data[var].dropna().values - data['t_trigger'][0])/1000

def get_mvt_reg(db_name, sub, sess):
    mov_dir = {'NAConf': op.join(paths.root_dir, paths.data_dir, "derivatives"),
               'EncodeProb': op.join(paths.root_dir, paths.data_dir, "derivatives"),
               'Explore': op.join(paths.root_dir, paths.data_dir, 'bids/derivatives/fmriprep-23.1.3_MAIN'), 
               'PNAS': op.join(paths.root_dir, paths.data_dir,
                               "MRI_data/raw_data")}

    # concatenate mvt_data across sessions_encode
    mvt_data = pd.DataFrame()

    if db_name == 'PNAS':

        fname = glob.glob(op.join(mov_dir[db_name],
                                    f"subj{sub:02d}",
                                    'fMRI',
                                    f"rp_aepi_sess{sess}_*.txt"))[0]
        
        mvt_data = pd.read_csv(fname, sep='\s+', header=None,
                            names=[f"mvt{k}" for k in range(6)])

    if db_name in ['EncodeProb', 'NAConf']:
        fname = glob.glob(op.join(mov_dir[db_name],
                            f'sub-{sub:02d}',
                                    f"rp_asub-{sub:02d}_task-*_run-0{sess}*.txt"))[0]
        mvt_data = pd.read_csv(fname, sep='\s+', header=None,
                            names=[f"mvt{k}" for k in range(6)])

    if db_name == 'Explore':
        fname = glob.glob(
            op.join(mov_dir[db_name], f"sub-{sub:02d}",
                    "func", f"sub-{sub:02d}_task-*_run-0{sess}*.tsv"))[0]

        mvt_data = pd.read_csv(fname, sep='\t')
        mvt_data = mvt_data[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
        mvt_data.columns = [f"mvt{k}" for k in range(6)]

    return mvt_data



