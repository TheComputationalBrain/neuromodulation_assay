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
from params_and_paths import *
from nilearn.input_data import MultiNiftiMasker

#from initializeSubject import initialize_subject

def get_json_dir(db_name, root_dir=root_dir, data_dir=data_dir):
   
    json_files_dir = {'NAConf': op.join(root_dir[data_access], data_dir['NAConf'], 'bids_dataset'),
                    'EncodeProb': op.join(root_dir[data_access], data_dir['EncodeProb'], 'bids_dataset'),
                    'Explore': '/home_local/EXPLORE/bids_dataset',
                    'PNAS': op.join(root_dir[data_access], data_dir['PNAS'], 'MRI_data/analyzed_data')}
    
    return json_files_dir[db_name]


def get_fmri_dir(db_name, root_dir=root_dir, data_dir=data_dir):
    
    fmri_dir = {'NAConf': op.join(root_dir[data_access], data_dir['NAConf'], 'derivatives'),
                'EncodeProb': op.join(root_dir[data_access], data_dir['EncodeProb'], 'derivatives'),
                'Explore': '/home_local/EXPLORE/DATA/bids/derivatives/fmriprep-22.1.1',
                'PNAS': op.join(root_dir[data_access], data_dir['PNAS'], 'MRI_data/analyzed_data')}

    return fmri_dir[db_name]

def get_beh_dir(db_name, root_dir=root_dir, data_dir=data_dir):

    beh_dir = {'NAConf': op.join(root_dir[data_access], data_dir['NAConf']),
               'EncodeProb': op.join(root_dir[data_access], data_dir['EncodeProb']),
               'Explore': '/home_local/EXPLORE/github/explore/2021_Continous_2armed',
               'PNAS': op.join(root_dir[data_access], data_dir['PNAS'])}

    return beh_dir[db_name]

def get_constants(data_dir, sub, sess):
    """This function looks for and imports constants data saved during an
    experiment for a given subject and session."""
    filepath = glob.glob(os.path.join(data_dir,
                                    'behavior',
                                    f'sub-{sub:02d}*',
                                    f'*session_{sess}.pickle'))[0]
    data = pickle.load(open(filepath, 'rb'))
    return data

def convert_to_secs(data, var):
    return (data[var].dropna().values - data['t_trigger'][0])/1000


def get_subjects(db, data_dir): 
    subjects = []
    if db == 'EncodeProb':
        folders = os.path.join(data_dir,
                                'behavior',
                                f'sub-*')
        
    for folder in os.listdir(folders):
        # Extract the number from the filename
        number = folder.split('sub_')[1]
        subjects.append(int(number))

    # #keep this as a test for now 
    # nsub_correct = {'PNAS': 21,
    #         'EncodeProb': 30,
    #         'NAConf': 60,
    #         'Explore': 60}
    
    # if len != nsub_correct[db]:
    #     raise ValueError(f"Found {nsub} subjects but experiment only contains {nsub_correct}.")

    return subjects

def get_seq(db, sub, sess, beh_dir):

    if db == 'Explore':
        if sub not in subnums_explore:
            subcsv = sub
        else :
            subcsv = subnums_explore[sub]

        file = os.path.join(beh_dir, 'Data_neurospin/formatted/fmri_session',
                            f'sub-{sub:02d}/data_subject_{subcsv:02d}_session_{sess}.csv')
        seq = pd.read_csv(file, index_col=0)

    else:
          
        if db == 'EncodeProb':
            if sub == 6:
                if sess > 1:
                    sess += 1
            if sub == 21:
                if sess == 4:
                    sess = 5

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

    filespath = os.path.join(data_dir,'Behavioral_data')
    beh_file = glob.glob(os.path.join(filespath,
                                      f'*SUBJECT_{sub}_Sess_{sess+1}_*'))
    if len(beh_file) > 1:
        raise('check behavioral files PNAS')


    beh_data = loadmat(beh_file[0])
    T0 = beh_data['save_T0'][0]
    indices_question = beh_data['StimQ'][0].astype(np.int)-1
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

def get_events_explore(subnum, sess, nan_missed=True):

    event_labels = ['cue', 'resp', 'out', 'missed', 'qA', 'qB', 'resp_qA', 'resp_qA', 'resp_qB', 'resp_qB']
    time_cols = ['trial_start', 'rt_start', 'outcome_start','missed_start',
                 'qAstart', 'qBstart', 'respA_val_start', 'respA_conf_start',
                 'respB_val_start', 'respB_conf_start']
    
    n_trials = 96
    
    trial_types_all = np.asarray(event_labels * n_trials)
    durations_all = np.asarray([0, 0, 0, 5, 0, 0, 0, 0, 0, 0] * n_trials)

    para = initialize_subject(subnum)# no need to specify vol because we only get onsets from this function
    para = para.set_index(np.arange(0, para.shape[0]))

    # Get onsets for missed trials
    para['missed_start'] = np.nan
    para.loc[para['ismissed'] == 1, 'missed_start'] = \
        para['trial_start'][para['ismissed'] == 1]

    # Don't count missed trials as regular trials
    if nan_missed:
        para.loc[para['ismissed'] == 1, 'trial_start'] = np.nan
        para.loc[para['ismissed'] == 1, 'outcome_start'] = np.nan
    para['rt_start'] = np.nansum(para[['trial_start', 'rt']], axis=1)
    para.loc[para['rt_start'] == 0, 'rt_start'] = np.nan

    para = para[para['block'] == (sess + 8)]

    #infer questions onsets
    
    para.loc[para['rtQ1_val'] == para['rtA_val'], 'Q1_is_A'] = 1 
    para.loc[para['rtQ1_val'] == para['rtB_val'], 'Q1_is_B'] = 1 
    ind_Q1_is_A = para[para['Q1_is_A'] == 1].index
    ind_Q1_is_B = para[para['Q1_is_B'] == 1].index

    for var in ['qAstart', 'qBstart', 'respA_val_start', 'respA_conf_start',
                'respB_val_start', 'respB_conf_start']:
        para[var] = np.nan
    
    for i in ind_Q1_is_A: 
        para['qAstart'][i] = para['trial_end'][i] + 1
        para['respA_val_start'][i] =  para['qAstart'][i] + para['rtA_val'][i]
        para['respA_conf_start'][i] = para['respA_val_start'][i] + 0.5 + para['rtA_conf'][i]
        para['qBstart'][i] = para['respA_conf_start'][i] + 1
        para['respB_val_startx'][i] = para['qBstart'][i] + para['rtB_val'][i]
        para['respB_conf_start'][i] = para['respB_val_start'][i] + 1 + para['rtB_conf'][i]

    for i in ind_Q1_is_B: 
        para['qBstart'][i] = para['trial_end'][i] + 1
        para['respB_val_start'][i] =  para['qBstart'][i] + para['rtB_val'][i]
        para['respB_conf_start'][i] = para['respB_val_start'][i] + 0.5 + para['rtB_conf'][i]
        para['qAstart'][i] = para['respB_conf_start'][i] + 1
        para['respA_val_start'][i] = para['qAstart'][i] + para['rtA_val'][i]
        para['respA_conf_start'][i] = para['respA_val_start'][i] + 1 + para['rtA_conf'][i]

    onsets = []
    onsets.append(para[time_cols].values)
    onsets = np.hstack(onsets).ravel('C')
    trial_types = trial_types_all[~np.isnan(onsets)]
    durations = durations_all[~np.isnan(onsets)]
    onsets = onsets[~np.isnan(onsets)]
    
    events = pd.DataFrame({'onset': onsets,
                         'trial_type': trial_types,
                         'duration': durations})

    return events

def get_sessions(db, sub, data_dir):

    #TODO: this probabaly needs to be adapted to the other experiments
    sessions = []
    filepath = os.path.join(data_dir,
                                    'behavior',
                                    f'sub-{sub:02d}/')
    for filename in os.listdir(filepath):
        if "sess_" in filename and filename.endswith(".csv"):
            # Extract the number from the filename
            number = filename.split("sess_")[1].split(".")[0]
            sessions.append(int(number))
        #TODO throw error if number appears more than once or if length over 4 (save 4 in the param file)
            
    # Check for duplicates
    if len(sessions) != len(set(sessions)):
        raise ValueError("Duplicate numbers found in the filenames.")

    # Check if the list contains more than 4 numbers -> this is just temporary?
    if len(sessions) > n_sessions:
        raise ValueError("Found more sessions than expected.")
       
    return sessions

def get_events(db, sub, sess, data_dir): 

    #TODO: the variables session and repsonse are currently only implemented for EncodeProb

    # # Adjust the session index based on the database and subject to handle specific cases
    # if db != 'EncodeProb':
    #     index = sess 
    # if db == 'EncodeProb':
    #     index = sess 
    #     if sub == 6:
    #         if sess > 1:
    #             index =  sess + 1
    #     elif sub == 21:
    #         if sess == 4:
    #             index =  sess + 1

    if db == 'PNAS': 

        data = get_data_PNAS(sub, sess, data_dir)
        onsets = np.hstack((data['stim_onset'], data['question_onset'],
                            data['response_onset']))
        duration = np.hstack(([0] * data['stim_onset'],
                            data['reaction_times'],
                            [0] * data['response_onset']))
        trial_type = np.hstack((['stim']*len(data['stim_onset']),
                                ['q']*len(data['question_onset']),
                                ['resp']*len(data['response_onset'])))

    if db != 'PNAS':

        if db == 'Explore':
            events = get_events_explore(sub, sess)
            return events

        else: 
            
            if db == 'EncodeProb':
                filepath = glob.glob(os.path.join(data_dir,
                                'behavior',
                                f'sub-{sub:02d}*',
                                f'*sess_{sess}.csv'))[0]
                data = pd.read_csv(filepath, header=1)
                on_stim = convert_to_secs(data, 't_on_stim')

                on_q_prob = convert_to_secs(data,'t_question_prob_on')
                on_q_conf = convert_to_secs(data,'t_question_conf_on')

                rt = np.append(data['estim_rt'].dropna().values/1000,
                            data['conf_rt'].dropna().values/1000)

                on_resp_prob = convert_to_secs(data,'t_rating_prob')
                on_resp_conf = convert_to_secs(data,'t_rating_conf')

                # off_q_prob = convert_to_secs(data, 't_question_prob_off')
                # off_q_conf = convert_to_secs(data, 't_question_conf_off')

                sub_prob = convert_to_secs(data, 'estim_position')
                sub_prob = rescale_answer(sub_prob, pos_min=-500,pos_max=500)
                sub_conf = convert_to_secs(data, 'conf_position')
                sub_conf = rescale_answer(sub_prob, pos_min=-500,pos_max=500)

                onsets = np.hstack((on_stim, on_q_prob, on_q_conf, on_resp_prob, on_resp_conf))
                trial_type = np.hstack((['stim'] * len(on_stim),
                                        ['q_prob'] * len(on_q_prob),
                                        ['q_conf'] * len(on_q_conf),
                                        ['resp_conf'] * len(on_resp_prob),
                                        ['resp_prob'] * len(on_resp_conf)))
                duration = np.hstack((np.zeros_like(on_stim),
                                    rt, np.zeros_like(on_resp_prob), np.zeros_like(on_resp_conf)))
                response = np.hstack((np.zeros_like(on_stim), sub_prob, sub_conf, 
                                      np.zeros_like(on_resp_prob), np.zeros_like(on_resp_conf))) 
                
            if db == 'NAConf':
                filespath = os.path.join(data_dir,
                        'behaviour_eyeTracker_data',
                        f'sub-{sub}')
                # pickle.load the experiment info file present in all subject behavioral folder
                file = os.path.join(filespath, f'experiment_info_sub-{sub:02d}.pickle')
                with open(file, 'rb') as f:
                        exp = pickle.load(f)
                # create event dataframe
                on_q = exp[sess-1]['question_onsets']
                resp = exp[sess-1]['response_onsets']

                onsets = np.hstack((exp[sess-1]['stim_onsets'],
                                    exp[sess-1]['question_onsets'],
                                    exp[sess-1]['response_onsets']))

                duration_stim = exp[sess-1]['durations']
                rt = exp[sess-1]['rt']
                resp_dur = [0] * len(resp)
                duration = np.hstack((duration_stim, rt, resp_dur)) #include both conf and probs rt
                trial_type = np.hstack((exp[sess-1]['conditions'],
                                        ['q']* len(on_q),
                                        ['resp']* len(resp)))
        
    events = pd.DataFrame({'onset': onsets,
                        'duration': duration,
                        'trial_type': trial_type,
                        'response': response
                        })

    return events

def convert_to_secs(data, var):
    return (data[var].dropna().values - data['t_trigger'][0])/1000

def get_mvt_reg(db_name, sub, sess, root_dir, data_dir):
    root_dir = root_dir[data_access]
    mov_dir = {'NAConf': op.join(root_dir, data_dir['NAConf'], "derivatives"),
               'EncodeProb': op.join(root_dir, data_dir['EncodeProb'], "derivatives"),
               'Explore': op.join(data_dir['Explore'], "mri_preproc"),
               'PNAS': op.join(root_dir, data_dir['PNAS'],
                               "MRI_data/analyzed_data")}

    # concatenate mvt_data across sessions
    mvt_data = pd.DataFrame()

    if db_name == 'PNAS':

        fname = glob.glob(op.join(mov_dir[db_name],
                                    f"subj{sub:02d}",
                                    'preprocEPI',
                                    f"rp_aepi_sess{sess}_*.txt"))[0]

    if db_name in ['EncodeProb', 'NAConf']:
        fname = glob.glob(op.join(mov_dir[db_name],
                            f'sub-{sub:02d}',
                                    f"rp_asub-{sub:02d}_task-*_run-0{sess}*.txt"))[0]

    if db_name == 'Explore':
        fname = glob.glob(
            op.join(mov_dir[db_name], f"sub-{sub:02d}",
                    "func", f"rp_asub-{sub:02d}_task-*_run-0{sess}*.txt"))[0]

    mvt_data = pd.read_csv(fname, sep='\s+', header=None,
                            names=[f"mvt{k}" for k in range(6)])

    return mvt_data



