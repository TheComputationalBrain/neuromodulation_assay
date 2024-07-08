#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 17:12:09 2022

@authors: Tiffany Bounmy, Florent Meyniel

Functions that create parametric modulations.
"""

import warnings
import numpy as np
import nilearn.signal
from nilearn.glm.first_level.hemodynamic_models import _hrf_kernel as hrf
from scipy.interpolate import interp1d
from params_and_paths import Params

params = Params()

def compute_cleaned_pmod_regs(onsets, durations, modulation_values, frame_times, tr):
    """
    Compute cleaned parametric modulation regressors with the given event onsets and durations
    and the given modulation values.
    This computation is done in three steps
    1. The modulation values are first centered.
    This has the effect of orthogonalizing the computed regressors with respect to
    the unmodulated event regressor in the GLM. As a result, the beta coefficient
    of the GLM corresponding to the unmodulated event regressor will model the
    mean of the BOLD signal when the event occurs.
    2. The event regressor (1 when event occurs, 0 otherwise) is multiplied by
    the (centered) modulation values and then convolved with the HRF.
    3. The same cleaning processing is applied to these regressors as to the BOLD signal.
    This is done to preserve the hypothesized linear relationship between
    the regressors and the BOLD signal after the signal has been processed.
    Parameters
    ----------
    onsets: np array of shape (n_events) with the onset of events (in seconds)
    durations: np array of shape (n_events) with the duration of each events (in second)
    modulation_values: 1D or 2D np array of shape (n_events) or (n_events, n_regs)
        with the parametric modulation values for each regressor (1 column = 1 regressor)
    frame_times: array of shape (n_scans)
    Returns
    -------
    regs: 2D np array of shape(n_scans, n_regs), the computed regressors.
    Notes
    -----
    We operate at the level of a session here: n_scans is the number of scans in one session.
    """
    # convert modulation_values into a 2D numpy array if needed
    if modulation_values.ndim == 1:
        modulation_values = modulation_values[:, np.newaxis]
    # center the modulation values
    modulation_values -= np.mean(modulation_values, axis=0)
    # compute the parametric modulation regressors convolved with the HRF
    regs = compute_regressor_fast(onsets, durations, modulation_values, frame_times)
    # clean the convolved regressors
    regs = clean_regs(regs, tr)

    return regs


def compute_regressor_fast(onsets, durations, values, frame_times):
    """
    This is the main function to convolve regressors with hrf model, adapted to improve speed from
    nilearn.glm.first_level.compute_regressor.
    Parameters
    ----------
    onsets: np array with the onset of events (in seconds)
    durations: np array with the duration of each events (in second)
    values: a np 2D array with the values of regressors, organized as (n_values, n_regressors)
    frame_times: array of shape (n_scans)
    NB, params (use below) is the parameter class prepared with settings, which contain:
        - the hrf model as hkernel
        - the oversampling factor to compute the regressor with high temporal resolution
    Returns
    -------
     computed regressors sampled at frame times, as an array of shape(n_scans, n_reg)
    Notes, in settings, the different hemodynamic models can be:
    'spm': this is the hrf model used in SPM
    'glover': this one corresponds to the Glover hrf
    It is expected that spm standard and Glover model would not yield
    large differences in most cases.
    """

    #? what is the signfificance of these parameters + are they default values
    dt = 0.125  # micro time resolution for fMRI modelling
    tr = frame_times[1] - frame_times[0]
    oversampling = int(tr/dt)
    hkernel = hrf(params.hrf, tr, oversampling=oversampling, fir_delays=None)

    # CREATE THE HIGH TEMPORAL RESOLUTION REGRESSORS
    # (inspired from _sample_condition)
    # Find the high-resolution frame_times
    n = frame_times.size

    # define a minimum onset (before this, events will be ignored)
    # this values is the defaut one in compute_regressor
    min_onset = float(-24)
    n_hr = ((n - 1) * 1. / (frame_times.max() - frame_times.min()) *
            (frame_times.max() * (1 + 1. / (n - 1)) - frame_times.min() -
             min_onset) * oversampling) + 1

    hr_frame_times = np.linspace(frame_times.min() + min_onset,
                                 frame_times.max() * (1 + 1. / (n - 1)), int(n_hr))

    if (onsets < frame_times[0] + min_onset).any():
        warnings.warn(('Some stimulus onsets are earlier than %s in the'
                       ' experiment and are thus not considered in the model'
                       % (frame_times[0] + min_onset)), UserWarning)

    # Set up the regressor timecourse
    tmax = len(hr_frame_times)
    regressor = np.zeros((len(hr_frame_times), values.shape[1]), dtype=float)
    t_onset = np.minimum(np.searchsorted(hr_frame_times, onsets), tmax - 1)
    regressor[t_onset, :] += values
    t_offset = np.minimum(
        np.searchsorted(hr_frame_times, onsets + durations),
        tmax - 1)

    # Handle the case where duration is 0 by offsetting at t + 1
    for i, t in enumerate(t_offset):
        if t < (tmax - 1) and t == t_onset[i]:
            t_offset[i] += 1

    regressor[t_offset, :] -= values
    hr_regressor = np.cumsum(regressor, axis=0)

    # CONVOLVE THE REGRESSORS AND HRF AND DOWNSAMPLE THE REGRESSOR
    conv_reg = np.vstack([[np.convolve(hr_reg, h)[:hr_reg.size] for h in hkernel]
                          for hr_reg in hr_regressor.T])

    # TEMPORALLY RESAMPLE THE REGRESSORS
    # (inspired by _resample_regressor)
    # NB: we do not orthogonolize the regressors
    f = interp1d(hr_frame_times, conv_reg)
    conv_reg = f(frame_times).T

    return conv_reg

def clean_regs(regs, tr):
    """
    Apply the same cleaning process to the given regressors as applied to the
    BOLD signal, i.e. first detrending and then high-pass filtering.
    """    
    return nilearn.signal.clean(regs,
        detrend=True,
        high_pass=params.hpf, t_r=tr,
        standardize=True
   )
