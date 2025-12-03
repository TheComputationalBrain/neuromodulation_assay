#this scripot defins only the prarams and paths that are needed to run the demo or analysise new data
#the original params_and_paths contains additonal info needed to located all the files in their original location and preprocess the data.

params = {
    'tasks': ['EncodeProb'],  #name of the task 
    'variables': ["Confidence"] #name of the variable to evaluate 
    }


paths = {
    'beta_dir': './data/beta_dir/study_1', #beta directory contains the beta maps as nifiti files (volume + voxel level)
    'results_dir': './results/study_1', #path at which to save all of the results
    'receptor_dir': './data/receptors' #path at which the receptor densitie dataframes are stored
}



receptors = {
    'receptor_names' : ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                        "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                        "MOR", "NET", "NMDA", "VAChT", "A2"],
    'serotonin' : ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    'acetylcholine' : ["A4B2", "M1", "VAChT"],
    'noradrenaline' : ["NET", "A2"],
    'opioid' : ["MOR"],
    'glutamate' : ["mGluR5", 'NMDA'],
    'histamine' : ["H3"],
    'gaba' : ["GABAa"],
    'dopamine' : ["D1", "D2", "DAT"],
    'cannabinnoid' : ["CB1"],
    'exc' : ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA'],
    'inh' : ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR', 'A2'],
    'receptor_label_formatted' : [
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
        '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$', '$5\\text{-}\\mathrm{HTT}$',
        '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$', '$\\mathrm{M}_{\\mathrm{1}}$',
        '$\\mathrm{VAChT}$', '$\\mathrm{NET}$', '$\\mathrm{A}_{\\mathrm{2}}$',
        '$\\mathrm{MOR}$', '$\\mathrm{mGluR}_{\\mathrm{5}}$', '$\\mathrm{NMDA}$',
        '$\\mathrm{H}_{\\mathrm{3}}$', '$\\mathrm{GABA}_{\\mathrm{a}}$', '$\\mathrm{D}_{\\mathrm{1}}$',
        '$\\mathrm{D}_{\\mathrm{2}}$', '$\\mathrm{DAT}$', '$\\mathrm{CB}_{\\mathrm{1}}$'
    ],
    'group_names'  : ['serotonin', 'acetylcholine', 'norepinephrine', 'opioid', 'glutamate', 'histamine', 'gaba', 'cannabinnoid']
    }

GROUP_KEYS = [
    "serotonin", "acetylcholine", "noradrenaline", "opioid",
    "glutamate", "histamine", "gaba", "dopamine", "cannabinnoid"
]

receptors["group_names"] = GROUP_KEYS
receptors["receptor_groups"] = [receptors[k] for k in GROUP_KEYS]
receptors["receptor_class"] = [receptors["exc"], receptors["inh"]]




