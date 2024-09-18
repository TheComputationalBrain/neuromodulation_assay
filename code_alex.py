#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:42:03 2024

@author: ap267379
"""

import os.path as op
import sys
sys.path.append('/home_local/EXPLORE/github/explore/2021_Continous_2armed/Analysis_python')

import numpy as np
import pandas as pd
import pickle

import fit_behavior_functions as fb
from idealObserver3 import io_with_derivations
# import behavior_ratings as br
import explore_initialize_constants as eic

# import fmri_firstlevel_functions as ff

ARM_IDS = eic.ARM_IDS
# DIR = eic.DIR
GENVOL = eic.GENVOL
BEHMODELDIR = eic.BEHMODELDIR

def get_ratings_with_io(subid, volmdlid, genvol=GENVOL, arm_ids=ARM_IDS,
                        which_io=["MAP_reward", "estimation_confidence"],
                        modeldir=None):
    
    # Constants
    IO_MAP = {"MAP_reward": "MAP", "estimation_confidence": "EC"}
    VOL_IDS = ["fit", "gen"]
    io_map = {long: short for long, short in IO_MAP.items() if long in which_io}
    iocols_base = [io_map[v] for v in io_map.keys()]
    respcols = ['value', 'conf']
    respcolsarm = [[f"opt{arm}_val" for arm in ARM_IDS], [f"opt{arm}_conf" for arm in ARM_IDS]]
    iocols = [f"{io}_{vol}" for io in iocols_base for vol in VOL_IDS]
    iocolsarm = [[f"{io}_{arm}_{vol}" for arm in ARM_IDS] for io in iocols_base for vol in VOL_IDS]
    gencolsarm = [[f"{arm}_mean" for arm in ARM_IDS]]
    gencols = ['mean']
    singlecols = ['subid', 'subuid', 'sessuid', 'sequid', 'quid', 'qnum'] + ['SD', 'isfree']
    
    # Get volatility for fitted vol version
    if modeldir is None:
        modeldir = DIR["model"]
    sub_modelpath = op.join(modeldir, subid, volmdlid + ".p")
    mdl = pickle.load(open(sub_modelpath, "rb"))
    volmdl_prms = pd.Series(mdl["params"], index=mdl["paramnames"], name="param")
    
    # Get data
    d = fb.get_data(subid)
    
    # Add ideal observer values (generative and fitted volatility)
    uid_vars = ["sessuid", "sequid"]
    d = fb.add_uid(d, uid_vars, "uid")
    iterids = d["uid"].unique()
    ioraw = {v: [] for v in VOL_IDS}
    io = {v: [] for v in VOL_IDS}
    
    for iid in iterids:
        di = d.loc[d["uid"] == iid]
        sequence = {"options": {arm_ids[0]: di["obsA"].values, 
                                arm_ids[1]: di["obsB"].values},
                    "outcome_SD_gen": di["SD"].values}
        ioraw["fit"] = io_with_derivations(sequence, volmdl_prms["vol"])
        ioraw["gen"] = io_with_derivations(sequence, genvol)
        for vol in io.keys():
            io[vol].append([])
            for iov, ion in io_map.items():
                io[vol][-1].append(pd.DataFrame(ioraw[vol][iov]))
                io[vol][-1][-1].index = di.index
                io[vol][-1][-1].columns = [f"{ion}_{arm}_{vol}" for arm in io[vol][-1][-1].columns]
                
            io[vol][-1] = pd.concat(io[vol][-1], axis=1)
    
    for vol in io.keys():
        io[vol] = pd.concat(io[vol], axis=0)
        d = pd.concat((d, io[vol]), axis=1)
    
    # Keep question trials only
    dq_ab = d.loc[d['isQ'] == 1, :]
    
    # Flatten columns across options A and B
    dq = {}
    narms = len(arm_ids)
    dq['armid'] = arm_ids * len(dq_ab)
    dq['armuid'] = [str(a+1) for a in range(narms)] * len(dq_ab)
    
    for v in singlecols:
        dq[v] = np.vstack([list(dq_ab[v].values)] * narms).T.ravel("C")
    
    for iv, v in enumerate(gencols):
        dq[v] = dq_ab[gencolsarm[iv]].values.ravel("C")
    
    for iv, v in enumerate(respcols):
        dq[v] = dq_ab[respcolsarm[iv]].values.ravel("C")
    
    for iv, v in enumerate(iocols):
        dq[v] = dq_ab[iocolsarm[iv]].values.ravel("C")
    
    dq = pd.DataFrame(dq)
    
    return dq


SUBIDS = [f'sub-{subnum:02d}' for subnum in eic.SUBNUMS]
subid = SUBIDS[0]
d = fb.get_data('sub-01')
d = d.loc[d['isfmri']].reset_index().drop(columns='index')

x_all = fb.parse_x(d, ['ER', 'EU', 'EUt', 'PE', 'UPE'])
volmodelid = fb.make_modelid('repeat', x_all=x_all, fit_vol=True, intercept=True)
# make_modelid(dv, x_all=None, fit_vol=False, fit_sd=False, intercept=True,

q_ratings = get_ratings_with_io(subid, volmodelid, modeldir=BEHMODELDIR)

# CU_CONS = ["ER", "EU"] # to be done better
# opt = {
#     "crossval": False,
#     "suffix": None,
#     "dm": {
#         "id": "cERc_cERu_cEUc_cERu_PE",                                                               # TO BE FILLED IN
#         "volmdl": "repeat_ER-EU-EUt-PE-cst-vol", # None=genvol; else modelid for beh model used
#         "zscore": False, 
#         "blocks": None,  # None=all blocks else list of block nums
#         "n_skipped": 4,
#         "confounds": [],                                                        # TO BE FILLED IN   #None=don't model in DM (implies they're in masker; else: list of which ones)
#         "confounds_src": "fmriprep-22.1.1", # Only if confounds are not none (shared with data)
#         "events": {
#             "id": ["cue", "out", "q", "missed"], #events to model (must be in events files)
#             "duration": None, # None=use durations from events file
#             "model_unmod": [True, True, False, True],
#             "split_free": [False, False, False, False], 
#             },
#         "modulators": {
#             "id": ["ER_chosen", "ER_unchosen", "EU_chosen", "EU_unchosen", "PE"],                                      # TO BE FILLED IN
#             "place": ["cue", "cue", "cue", "cue", "out"],                                        # TO BE FILLED IN
#             "split_free": [True, True, True, True, True],
#             "model_forced": [True, True, True, True, True], 
#             "version": "0"  # 0=zero-centered, z=scored, None=normal units
#             }
#         },
#     "data": {
#         "prep": "fmriprep-22.1.1",  # "custom-2.0" "fmriprep-22.1.1"
#         "space": "norm", # norm or native
#         "mask_type": "indiv",  # indiv or group (can only be indiv in native space)
#         "mask": WHICH_MASK, # brain, GM
#         "confounds": [],                                                        # TO BE FILLED IN
#         "confounds_space": "native",
#         "smooth": FIRSTLEVEL_SMOOTH, 
#         "high_pass": 1/128, 
#         "detrend": True,
#         "zscore": True
#         }
#     }


# Function for making firstlevel design matrices

# ff.make_design_matrix(sub, behdir, opt)
# which_variables=['ER', 'EC', 'PE']
which_variables = ['expected_reward', 'estimation_confidence', 'prediction_error']

# NB RUN THIS PER BLOCK
a = io_with_derivations({'options': {'A': d['obsA'].values, 'B': d['obsB'].values}, 'outcome_SD_gen': d['SD'].values}, 
                        vol=1/24, which_variables=which_variables, as_predictors=True) # < the output will contain not just ER and EU but also the prior which you can use to compute entropy


#                  split=None, suffix='')
# fb.make_
# br.get_ratings_with_io(subid, volmdlid, genvol=GENVOL, arm_ids=ARM_IDS,
#                         which_io=["MAP_reward", "estimation_confidence"],
#                         modeldir=None)