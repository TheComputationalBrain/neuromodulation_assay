import os

def load_paths(task=None):

    base = {
        "home_dir": "/home_local/alice_hodapp/NeuroModAssay",
        "root_dir": "/neurospin/unicog/protocols/IRMf",
        "pet_path": "/home/ah278717/hansen_receptors/data/PET_nifti_images/",
        "alpha_path": "/home/ah278717/alpha2_receptor/",
        "out_dir": "/home_local/alice_hodapp/NeuroModAssay/results",
    }

    # NEW: receptor path
    base["receptor_dir"] = os.path.join(base["home_dir"], "regressor")

    # If "all" → return *only* base settings
    if task == "all" or task is None:
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

    if task not in task_dirs:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {list(task_dirs)}")

    # Combine base + task-specific paths
    paths = base.copy()
    paths["data_dir"] = os.path.join(
        base["root_dir"],
        task_dirs[task]["data_dir"]
    )
    paths["mov_dir"] = os.path.join(
        paths["data_dir"],
        task_dirs[task]["mov_dir"]
    )

    return paths
