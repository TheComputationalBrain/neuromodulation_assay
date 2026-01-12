PARCELATED = False
UPDATE_REG = False   


def load_params(task, cv_true=False):
    """Return a dictionary of parameters for the given task."""

    # Shared defaults
    base = {
        "mask": "schaefer",
        "parcelated": PARCELATED,
        "update": UPDATE_REG,
        "hpf": 1/128,
        "hrf": "spm",
        "zscore_per_session": True,
        "study_mapping": {
            "EncodeProb": "Study 1",
            "NAConf": "Study 2",
            "PNAS": "Study 3",
            "Explore": "Study 4",
        },
        "latent_vars": ["confidence", "surprise"],
        "variables_long": ["surprise", "confidence", "surprise_neg", "confidence_neg"],
        'tasks' : ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
    }

    # If "all" → return *only* base settings
    if task == "all":
        return base

    # Task-specific configurations
    configs = {

        "language": {
            "tasks": ["lanA"],
            "ignore": [80],
            "latent_vars": ["S-N"],
        },

        "EncodeProb": {
            "db": "EncodeProb",
            "seq_type": "bernoulli",
            "smoothing_fwhm": 5,
            "ignore": [1, 4, 12, 20],
            "session": {6: [1, 3, 4, 5], 20: [1, 2, 3, 5], 21: [1, 2, 3, 5]},
            "io_options": {"p_c": 1/75, "resol": 20}
        },

        "NAConf": {
            "db": "NAConf",
            "seq_type": "bernoulli",
            "smoothing_fwhm": 5,
            "naconf_behav_subj": [
                1, 2, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 55, 56,
                57, 58, 59, 60, 61
            ],
            "ignore": [3, 5, 6, 9, 36, 51] if not cv_true else
                       [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59],
            "session": [],
            "io_options": {"p_c": 1/75, "resol": 20}
        },

        "PNAS": {
            "db": "PNAS",
            "seq_type": "transition",
            "smoothing_fwhm": 5,
            "ignore": [],
            "session": [],
            "io_options": {"p_c": 1/75, "resol": 20},
        },

        "Explore": {
            "db": "Explore",
            "smoothing_fwhm": 5,
            "ignore": [9, 17, 46],
            "session": [],
            "split": False,
            "reward": False,
            "io_variables": ["US", "EC_chosen"],
        },
    }

    # Merge and return
    return {**base, **configs[task]}

