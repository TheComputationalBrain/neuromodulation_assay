from config.utils import AttrDict

def load_paths(task = None):

    base = {
        "home_dir": "/home_local/alice_hodapp/NeuroModAssay",
        "root_dir": "/neurospin/unicog/protocols/IRMf",
        "receptor_path": "/home/ah278717/hansen_receptors/data/PET_nifti_images/",
        "alpha_path": "/home/ah278717/alpha2_receptor/",
        "out_dir": "/home_local/alice_hodapp/NeuroModAssay/results"
    }

    # If "all" → return *only* base settings
    if task == "all":
        return base
    
    task_dirs = {
        "NAConf": {
            "data_dir": "MeynielMazancieux_NACONF_prob_2021",
            "mov_dir": "derivatives"
        },
        "EncodeProb": {
            "data_dir": "EncodeProb_BounmyMeyniel_2020",
            "mov_dir": "derivatives"
        },
        "Explore": {
            "data_dir": "Explore_Meyniel_Paunov_2021",
            "mov_dir": "bids/derivatives/fmriprep-23.1.3_MAIN"
        },
        "PNAS": {
            "data_dir": "Meyniel_MarkovGuess_2014",
            "mov_dir": "MRI_data/raw_data"
        }
    }


    return {**base, **task_dirs[task]}
