#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Florent Meyniel, Audrey Mazancieux

This script performs the second-level analysis for the probability learning task of 
the NACONF dataset.
It uses the output of the first-level analysis, compute stats, and plot results.

The code has been adapted to work with the Neuromod project

"""

import os
import pickle
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from params_and_paths import Paths, Params

paths = Paths()
params = Params()

OUT_DIR = '/home_local/alice_hodapp/NeuroModAssay/domain_general/behavior'

N_BINS = 6
FONTSIZE = 16

with open(os.path.join(OUT_DIR, 'correlations_IO_all_sub.pickle'), 'rb') as f:
    corr_dict = pickle.load(f)

with open(os.path.join(OUT_DIR, 'NAConf_behav_summary.txt'), "w") as file:

    # group-level test
    for name in ['r_prob', 'r_conf', 'res_r_conf']:
        mean = np.mean(np.fromiter(corr_dict[name].values(), dtype=float))
        std = np.std(np.fromiter(corr_dict[name].values(), dtype=float))
        ttest = ttest_1samp(np.fromiter(corr_dict[name].values(), dtype=float), 0)
        file.write(f"{name}: mean={mean}, SD={std}, t={ttest.statistic}, df={ttest.df}, p={ttest.pvalue}")
        file.write('\n\n')
        plt.figure()
        plt.plot(np.zeros(len(np.fromiter(corr_dict[name].values(), dtype=float))),
                np.fromiter(corr_dict[name].values(), dtype=float), '.')
        plt.ylabel(name)
        fname = f'NAConf_behav_{name}.png' 
        plt.savefig(os.path.join(OUT_DIR, fname))

# # reformat group data into a panda dataframe
# with open(os.path.join(OUT_DIR, 'behav_io_data_all_sub.pickle'), 'rb') as f:
#     behav_io_data_all_sub = pickle.load(f)

# subjects = behav_io_data_all_sub['sub_prob'].keys()
# data = pd.DataFrame(
#     columns=['subject', 'sub_pred', 'sub_conf', 'res_sub_conf', 'io_pred', 'io_conf',
#              'io_surp', 'io_entropy'], dtype='float64')
# for subject_id in subjects:
#     # Put into a panda data frame
#     n_questions = len(behav_io_data_all_sub['sub_prob'][subject_id])
#     sub_data = pd.DataFrame({
#         'subject': [subject_id]*n_questions,
#         'sub_pred': behav_io_data_all_sub['sub_prob'][subject_id],
#         'sub_conf': behav_io_data_all_sub['sub_conf'][subject_id],
#         'res_sub_conf': behav_io_data_all_sub['sub_res_conf'][subject_id].flatten(),
#         'io_pred': behav_io_data_all_sub['io_prob'][subject_id],
#         'io_conf': behav_io_data_all_sub['io_conf'][subject_id],
#         'io_surp': behav_io_data_all_sub['io_surp'][subject_id],
#         'io_entropy': behav_io_data_all_sub['io_entr'][subject_id],
#         })

#     # concatenate
#     data = pd.concat([data, sub_data])

# # remove missed trials
# data.dropna(inplace=True)

# # Compute within subject correlations and linear regressions
# subj_correlations = pd.DataFrame(
#     index=subjects,
#     columns=['sub_io_conf', 'sub_io_conf_int', 'sub_io_conf_slope',
#              'sub_io_pred', 'sub_io_pred_int', 'sub_io_pred_slope',
#              'res_sub_io_conf', 'res_sub_io_conf_int',
#              'res_sub_io_conf_slope'])
# for subject in subjects:
#     sub_data = data[data['subject'] == subject]
#     for (x, y, var) in zip(['io', 'io', 'io'],
#                            ['sub', 'res_sub', 'sub'],
#                            ['conf', 'conf', 'pred']):

#         # correlation
#         subj_correlations.loc[subject]["_".join([y, x, var])] = \
#             np.corrcoef(sub_data["_".join([x, var])], sub_data["_".join([y, var])])[0, 1]

#         # linear regression
#         reg = LinearRegression().fit(sub_data["_".join([x, var])].values.reshape(-1, 1),
#                                      sub_data["_".join([y, var])])
#         subj_correlations.loc[subject]["_".join([y, x, var, 'int'])] = reg.intercept_
#         subj_correlations.loc[subject]["_".join([y, x, var, 'slope'])] = reg.coef_[0]

# # compute correlations between different variables
# subj_correlations_tmp = pd.DataFrame(
#     index=subjects,
#     columns=['sub_conf_io_surp', 'sub_conf_io_surp_int', 'sub_conf_io_surp_slope',
#              'sub_conf_io_entropy', 'sub_conf_io_entropy_int', 'sub_conf_io_entropy_slope',
#              'io_conf_io_surp', 'io_conf_io_surp_int', 'io_conf_io_surp_slope',
#              'io_conf_io_entropy', 'io_conf_io_entropy_int', 'io_conf_io_entropy_slope'])

# for subject in subjects:
#     sub_data = data[data['subject'] == subject]
#     for (x, y, var_x, var_y) in zip(['io', 'io', 'io', 'io'],
#                                     ['sub', 'sub', 'io', 'io'],
#                                     ['surp', 'entropy', 'surp', 'entropy'],
#                                     ['conf', 'conf', 'conf', 'conf']):

#         # correlation
#         subj_correlations_tmp.loc[subject]["_".join([y, var_y, x, var_x])] = \
#             np.corrcoef(sub_data["_".join([x, var_x])], sub_data["_".join([y, var_y])])[0, 1]

#         # linear regression
#         reg = LinearRegression().fit(sub_data["_".join([x, var_x])].values.reshape(-1, 1),
#                                      sub_data["_".join([y, var_y])])
#         subj_correlations_tmp.loc[subject]["_".join([y, var_y, x, var_x, 'int'])] = reg.intercept_
#         subj_correlations_tmp.loc[subject]["_".join([y, var_y, x, var_x, 'slope'])] = reg.coef_[0]

# subj_correlations = pd.concat([subj_correlations, subj_correlations_tmp], axis=1)

# subj_correlations.to_csv(os.path.join(OUT_DIR, 'NACONF_subj_correlations.csv'))


# Test selected correlation for significance on group level


# Plot group level effects, binned, with linear fit

# analysis_list = [
#     {'dep_var': 'sub_pred', 'ind_var': 'io_pred', 'reg': 'sub_io_pred',
#      'x_label': 'Ideal probability estimate', 'y_label': 'Subjective probability estimate'},
#     {'dep_var': 'sub_conf', 'ind_var': 'io_conf', 'reg': 'sub_io_conf',
#      'x_label': 'Ideal confidence \n(log precision)', 'y_label': 'Subjective confidence'},
#     {'dep_var': 'res_sub_conf', 'ind_var': 'io_conf', 'reg': 'res_sub_io_conf',
#      'x_label': 'Ideal confidence \n(log precision)', 'y_label': 'Residual subjective confidence'},
#     {'dep_var': 'sub_conf', 'ind_var': 'io_surp', 'reg': 'sub_conf_io_surp',
#      'x_label': 'Ideal surprise \n on the preceding trial', 'y_label': 'Subjective confidence'},
#     {'dep_var': 'sub_conf', 'ind_var': 'io_entropy', 'reg': 'sub_conf_io_entropy',
#      'x_label': 'Unpredictability', 'y_label': 'Subjective confidence'},
#     {'dep_var': 'io_conf', 'ind_var': 'io_surp', 'reg': 'io_conf_io_surp',
#      'x_label': 'Ideal surprise \n at the preceding trial', 'y_label': 'Ideal confidence'},
#     {'dep_var': 'io_conf', 'ind_var': 'io_entropy', 'reg': 'io_conf_io_entropy',
#      'x_label': 'Unpredictability', 'y_label': 'Ideal confidence'}]

# for analysis in analysis_list:
#     binned_data = data.groupby(by=[pd.qcut(data[analysis['ind_var']], N_BINS), 'subject']).mean()
#     binned_data = binned_data.groupby(level=analysis['ind_var'])
#     fig, ax = plt.subplots(1, 1, figsize=(4, 4))
#     if analysis['dep_var'] == 'io_pred':
#         xlim = [0, 1]
#     else:
#         xlim = np.array([binned_data.mean()[analysis['ind_var']].min() - 0.1,
#                          binned_data.mean()[analysis['ind_var']].max() + 0.1])
#     if analysis['dep_var'] == 'sub_conf':
#         ls = '--'
#     else:
#         ls = '-'
#     ax.plot(xlim,
#              xlim * subj_correlations[analysis['reg'] + '_slope'].mean() +
#              subj_correlations[analysis['reg'] + '_int'].mean(),
#              lw=3,
#              color="darkgrey",
#              zorder=1,
#              ls=ls)
#     ax.errorbar(binned_data.mean()[analysis['ind_var']],
#                  binned_data.mean()[analysis['dep_var']],
#                  binned_data.sem()[analysis['dep_var']],
#                  fmt='o', capsize=8,
#                  markersize=8,
#                  color="black",
#                  zorder=2)
#     ax.set_xlabel(analysis['x_label'], fontsize=FONTSIZE-2)
#     ax.set_ylabel(analysis['y_label'], fontsize=FONTSIZE-2)
#     ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-4, pad=8)
#     fig.tight_layout()


