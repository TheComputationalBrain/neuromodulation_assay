#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:03:17 2022

# TO ADD:
# questions data frame

@author: apaunov
"""

# Imports
import numpy as np
from glob import glob
import os.path as op
import pandas as pd
import pickle
import warnings

from preprocessing.idealObserver3 import io_with_derivations

# Helper functions
def zscore(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def zero_center(x):
    return x - np.nanmean(x)


# Main function
def initialize_subject(sub, datadir=None, vol=4/96, window_size=2, as_predictors=True):
    # DATADIR = '../Data_neurospin/formatted'
    subdir = f'/home_local/EXPLORE/DATA/bids/raw/sub-{sub:02d}/func/'

    # choiceB, obs are for check: use the ones from IO if identical
    data_fields = ['trial', 'block', 'seqSet', 'seqNumSet', 'seqNum',
                   'seqNumDalin', 'choiceA', 'choiceB', 'choiceR', 'colorP',
                   'rt', 'reward', 'wrong', 'ismissed', 'keyMissB',
                   'rtMiss', 'trial_start', 'outcome_start', 'trial_dur',
                   'outcome_dur', 'SOA', 'ITI_actual', 'A', 'B', 'SD',
                   'forced', 'isfree', 'A_mean', 'B_mean', 'A_cp', 'B_cp',
                   'Left', 'Right', 'forcedLeft', 'Left_mean', 'Right_mean',
                   'Left_cp', 'Right_cp', 'isQ', 'trial_end', 'rtQ1_val',
                   'rtQ1_conf', 'rtQ2_val', 'rtQ2_conf', 'rtA_val',
                   'rtA_conf', 'rtB_val', 'rtB_conf', 'optA_val', 'optA_conf', 'optB_val', 'optB_conf'] 

    # columns specific to fMRI data
    # (all of them, even though only some are carried over here)
    fmri_cols = ['keyMissA', 'keyMiss', 'ismissed', 'ITI_actual', 'rtMiss',
                 'keyMissR', 'keyMissB', 'trial_start_norm', 'keyMissL']
    n_fmri_cols = len(fmri_cols)
    n_trials = 96
    # N_BLOCKS = [8, 4]
    N_BLOCKS = 4
    # nan_cols = pd.DataFrame(data=np.nan * np.ones((n_trials, n_fmri_cols)),
    #                         columns=fmri_cols)

    n_free_seg = 8  # number of free-forced alternations per block


    free_forced_idx = np.asarray(
        [np.arange(1, n_free_seg + 1)] * int(n_trials / n_free_seg)).ravel('F')

    arm_ids = ['A', 'B']

    IO_MAP = {
        'options': 'obs',
        'MAP_reward': 'RMAP',
        'expected_reward': 'ER',
        'expected_discrete_reward': 'ERD',
        'expected_uncertainty': 'EU',
        'expected_outcome_uncertainty': 'EUO',
        'unexpected_uncertainty': 'UU',
        'prediction_error': 'PE',
        'feedback_surprise': 'US',
        'signed_feedback_surprise': 'SS',
        'unsigned_prediction_error': 'UPE',
        'estimation_confidence': 'EC',
        'entropy': 'entropy'
        }

    df = []
    iblock = 0
    # for sess in sess_types:
    #     sess_path = op.join(datadir, sess + '_session', sub)
        # if op.exists(sess_path):
    f_paths = sorted(glob(op.join(subdir, '*_beh.tsv')))
    for f in f_paths:
        data = pd.read_csv(f, sep="\t")

        df.append(pd.DataFrame())
        for col in data_fields:
            df[iblock][col] = data[col].values.astype(float)

        df[iblock]['free_seg_num'] = free_forced_idx

        sequence = {'options': {'A': data['obsA'].values,
                                'B': data['obsB'].values},
                    'outcome_SD_gen': data['SD'].values}

        io = io_with_derivations(sequence, vol=vol,
                                  which_variables=['all'],
                                  as_predictors=as_predictors,
                                  reward_levels=(30, 50, 70),
                                  reward_range=(1, 100),
                                  window_size=window_size)
        
        reward_levels = [30,50,70]

        for io_name, df_name in IO_MAP.items():
            for arm in arm_ids:
                df[iblock][df_name + '_' + arm] = \
                    np.asarray(io[io_name][arm]).astype(float)

        iblock = iblock + 1

    df = pd.concat(df, ignore_index=True)
    n_obs = df.shape[0]

    # Add column for fMRI block count (as 1-4 rather than 9-12)
    # df['block_fmri'] = np.nan * np.ones(n_obs)
    # for iblock in range(N_BLOCKS[1]):
    #     blocknum_fmri = iblock + 1
    #     blocknum = blocknum_fmri + N_BLOCKS[0]
    #     df.loc[df['block'] == blocknum, 'block_fmri'] = blocknum_fmri

    # Add flags for first trial after forced choice
    df['isfree_and_first'] = (
        np.hstack((0, np.diff(df['isfree']))) == 1).astype(float)
    df['isfree_and_not_first'] = ((df['isfree_and_first'] == 0)
                                  & (df['isfree'] == 1)).astype(float)

    # Convert time variables to seconds
    time_cols = ['rt', 'rtMiss', 'trial_start', 'outcome_start',
                 'trial_dur', 'outcome_dur', 'SOA', 'ITI_actual', 
                 'trial_end', 'rtQ1_val','rtQ1_conf', 'rtQ2_val', 'rtQ2_conf',
                 'rtA_val','rtA_conf', 'rtB_val', 'rtB_conf']
    for col in time_cols:
        df[col] = df[col] / 1000

    df['q_start'] = np.nan * np.ones(n_obs)
    df.loc[df['isQ'] == 1, 'q_start'] = \
        np.sum(df.loc[df['isQ'] == 1, ['trial_start', 'trial_dur']], axis=1)

    # Get pred. error, surprise (signed and unsigned) on chosen arm
    # (they're undefined for the unchosen one)
    obs_only_cols = ['PE', 'US', 'SS', 'UPE']
    for col in obs_only_cols:
        armcols = [f"{col}_A", f"{col}_B"]
        for armcol in armcols:
            df.loc[np.isnan(df[armcol]), armcol] = 0        
        df[col] = np.nansum(df[armcols], axis=1)
        df[col + '_diff'] = np.diff(df[armcols], axis=1)

    # Get previous RPE, SU, SS:
    for col in obs_only_cols:
        tmp = []
        for block in np.unique(df['block']).astype(int):
            tmp.append(np.hstack((0, df[col][df['block'] == block])))
            tmp[block-1] = tmp[block-1][:-1]
        tmp = np.hstack(tmp)
        df[col + '_prev'] = tmp
        
        # prev diff
        tmp = []
        for block in np.unique(df['block']).astype(int):
            tmp.append(np.hstack((0, df[col + '_diff'][df['block'] == block])))
            tmp[block-1] = tmp[block-1][:-1]
        tmp = np.hstack(tmp)
        df[col + '_diff_prev'] = tmp

    # Code choices as repeat=1, switch=0
    # (for missed trials: count both missing trial and the one after it as nan)
    repeat = []
    for block in np.unique(df['block']).astype(int):
        choices = df['choiceB'][df['block'] == block]
        repeat.append([np.nan] +
                      [current == previous for (previous, current) in
                       zip(choices[0:-1], choices[1:])])
    df['repeat'] = np.hstack(repeat).astype(float)
    df.loc[df['ismissed'] == 1, 'repeat'] = np.nan
    df.loc[np.hstack((0, df['ismissed'][0:-1].values)) == 1, 'repeat'] = np.nan
    
    # ADDED
    df['repeat_sign'] = df['repeat'].copy()
    df.loc[df['repeat'] == 0, 'repeat_sign'] = -1
    df["PE_prevrep"] = df["PE_prev"] * df["repeat_sign"]
    
    # Get difference between arms, the average, and the chosen arm
    AB_cols = ['RMAP', 'ER', 'ERD', 'EU', 'EUO', 'UU', 'EC', 'entropy']
    for col in AB_cols:
        df[col + '_diff'] = np.diff(df[[col + '_A', col + '_B']], axis=1)
        df[col + '_abs_diff'] = abs(df[col + '_diff'])
        df[col + '_mean'] = np.nanmean(df[[col + '_A', col + '_B']], axis=1)
        df[col + '_chosen'] = np.nan * np.ones(n_obs)
        df[col + '_unchosen'] = np.nan * np.ones(n_obs)
        for iarm, arm in enumerate(arm_ids):
            df.loc[df['choiceB'] == iarm, col + '_chosen'] = \
                df[col + '_' + arm][df['choiceB'] == iarm]
            df.loc[df['choiceA'] == iarm, col + '_unchosen'] = \
                df[col + '_' + arm][df['choiceA'] == iarm]
        df[col + '_diffcu'] = np.diff(df[[col + '_unchosen', col + '_chosen']],
                                      axis=1)
    # Get difference between repeat and non-repeat arms
    stay_option = [np.nan] + list(df['choiceB'][0:-1])
    switch_option = [np.nan] + list(-df['choiceB'][0:-1] + 1)
    for col in AB_cols:
        df[f'{col}_diffrs'] = [np.nan if np.isnan(stay)
                               else value[int(stay)] - value[int(switch)]
                               for (stay, switch, value)
                               in zip(stay_option, switch_option,
                                      df[[f'{col}_A', f'{col}_B']].values)]

    # Apply transformations to columns of interest
    col_contains = ['RMAP', 'ER', 'ERD', 'EU', 'EUO', 'UU', 'PE', 'US', 'SS', 'UPE', 'EC','entropy']
    to_transform = []
    for col in list(df.columns):
        if np.any([col_str in col for col_str in col_contains]):
            to_transform.append(col)
    to_transform = to_transform + ['reward']  # add other columns to z-score
    for col in to_transform:
        df = pd.concat((df, pd.Series.rename(zscore(df[col]),
                                             index=col + '_z')), #, axis=1)
                       axis=1)
        df = pd.concat((df, pd.Series.rename(zero_center(df[col]),
                                             index=col + '_0')),  # , axis=1
                       axis=1)

    # Trial classification
    df = df.copy()
    df["exploit"] = (df['ER_diffcu'] > 0).astype(int)
    df["explore"] = (df['EU_diffcu'] > 0).astype(int)
    df["exploit_pure"] =  ((df["exploit"] == 1) & (df["explore"] == 0)).astype(int)
    df["explore_pure"] =  ((df["exploit"] == 0) & (df["explore"] == 1)).astype(int)
    df["ws"] = (df["PE_prev"] > 0) & (df["repeat"] == 1).astype(int)
    df["ls"] = (df["PE_prev"] < 0) & (df["repeat"] == 0).astype(int)
    df["wsn"] = (df["PE_prev"] > 0) & (df["repeat"] == 0).astype(int)
    df["lsn"] = (df["PE_prev"] < 0) & (df["repeat"] == 1).astype(int)
    df["wsls"] = (((df["PE_prev"] > 0) & (df["repeat"] == 1)) | 
                  ((df["PE_prev"] < 0) & (df["repeat"] == 0))).astype(int)
    df["wslsn"] = (((df["PE_prev"] > 0) & (df["repeat"] == 0)) | 
                  ((df["PE_prev"] < 0) & (df["repeat"] == 1))).astype(int)
    
    # # V1: non-exclusive categories
    # df["ER_chosen_exploit_0"] = df["ER_chosen_0"] * df["exploit"]
    # df["ER_chosen_nonexploit_0"] = df["ER_chosen_0"] * np.abs(1-df["exploit"])
    # df["EU_chosen_explore_0"] = df["EU_chosen_0"] * df["explore"]
    # df["EU_chosen_nonexplore_0"] = df["EU_chosen_0"] * np.abs(1-df["explore"])
    
    # V2: exclusive categories
    df["ER_chosen_exploit_0"] = df["ER_chosen_0"] * df["exploit"]
    df["ER_chosen_explore_0"] = df["ER_chosen_0"] * np.abs(1-df["exploit"])
    df["EU_chosen_exploit_0"] = df["EU_chosen_0"] * df["exploit"]
    df["EU_chosen_explore_0"] = df["EU_chosen_0"] * np.abs(1-df["exploit"])
    
    # PE_PREVREP PER SD LEVEL
    df["highsd"] = (df['SD'] > 10).astype(int)
    df["PE_prevrep_highsd_0"] = df["PE_prevrep_0"] * df["highsd"]
    df["PE_prevrep_lowsd_0"] = df["PE_prevrep_0"] * np.abs(1-df["highsd"])
    
    df["wsls_highsd"] = df["wsls"] * df["highsd"]
    df["wsls_lowsd"] = df["wsls"] * np.abs(1-df["highsd"])
    
    df["ws_highsd"] = df["ws"] * df["highsd"]
    df["ws_lowsd"] = df["ws"] * np.abs(1-df["highsd"])
    
    df["ls_highsd"] = df["ls"] * df["highsd"]
    df["ls_lowsd"] = df["ls"] * np.abs(1-df["highsd"])
    
    df["PE_prevrep_EU_mean_0"] = df["PE_prevrep_0"] * df["EU_mean_0"]
    df["wsls_EU_mean_0"] = df["wsls"] * df["EU_mean_0"]
    df["ws_EU_mean_0"] = df["ws"] * df["EU_mean_0"]
    df["ls_EU_mean_0"] = df["ls"] * df["EU_mean_0"]
    # (NB: not accounting for missed trials in fMRI)
    # df = pd.concat(
    #     (df, pd.Series((df['ER_diffcu'] > 0).values, name='high_ER')), axis=1)
    # df = pd.concat(
    #     (df, pd.Series((df['EU_diffcu'] > 0).values, name='high_EU')), axis=1)
    # df = pd.concat(
    #     (df, pd.Series((df['UU_diffcu'] > 0).values, name='high_UU')), axis=1)

    # df = pd.concat((df, pd.Series(
    #     (df['high_ER'] & ~(df['high_EU'] | df['high_UU'])).values,
    #     name='exploit')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     (df['high_EU'] & ~(df['high_ER'] | df['high_UU'])).values,
    #     name='explore_EU')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     (df['high_UU'] & ~(df['high_ER'] | df['high_EU'])).values,
    #     name='explore_UU')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     ((df['high_EU'] & df['high_UU']) & ~df['high_ER']).values,
    #     name='explore_Both')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     ((df['high_ER'] & df['high_EU']) & ~df['high_UU']).values,
    #     name='ER_and_EU')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     ((df['high_ER'] & df['high_UU']) & ~df['high_EU']).values,
    #     name='ER_and_UU')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     (df['high_ER'] & df['high_UU'] & df['high_EU']).values,
    #     name='all')), axis=1)
    # df = pd.concat((df, pd.Series(
    #     ~(df['high_ER'] | df['high_UU'] | df['high_EU']).values,
    #     name='none')), axis=1)

    # df['exploit'] = (df['high_ER'] & ~(df['high_EU'] | df['high_UU']))
    # df['explore_EU'] = (df['high_EU'] & ~(df['high_ER'] | df['high_UU']))
    # df['explore_UU'] = (df['high_UU'] & ~(df['high_ER'] | df['high_EU']))
    # df['explore_Both'] = ((df['high_EU'] & df['high_UU']) & ~df['high_ER'])
    # df['ER_and_EU'] = ((df['high_ER'] & df['high_EU']) & ~df['high_UU'])
    # df['ER_and_UU'] = ((df['high_ER'] & df['high_UU']) & ~df['high_EU'])
    # df['all'] = (df['high_ER'] & df['high_UU'] & df['high_EU'])
    # df['none'] = ~(df['high_ER'] | df['high_UU'] | df['high_EU'])

    # Defragment
    df = df.copy()

    return df

# With fitted volatility
def initialize_subject_fitvol(sub, datadir, mdlid, voldir=None, window_size=2, as_predictors=True):
    "datadir is the bids directory (used for? in the func above)"
    "voldir is the behavioral models directory"
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if voldir is None:
        voldir = "/home_local/EXPLORE/DATA/beh_model/Behavioral_models"
    
    volpath = op.join(voldir, sub, f"{mdlid}.p")
    mdl = pickle.load(open(volpath, "rb"))
    names, prms = mdl["paramnames"], mdl["params"]
    vol = [p for nm, p in zip(names, prms) if nm == "vol"][0]
    io = initialize_subject(sub, datadir, vol=vol, 
                            window_size=window_size, as_predictors=as_predictors)
    
    return io

