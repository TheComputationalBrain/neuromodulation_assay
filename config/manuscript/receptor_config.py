def load_receptors(source="PET2"):
    if source not in ["PET", "PET2"]:
        raise ValueError("source must be 'PET' or 'PET2'")

    if source == "PET":
        receptor_names = ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT",
                          "A4B2","CB1","D1","D2","DAT","GABAa","H3","M1",
                          "mGluR5","MOR","NET","NMDA","VAChT"]

        noradrenaline = ["NET"]
    else:  # PET2
        receptor_names = ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT",
                          "A4B2","CB1","D1","D2","DAT","GABAa","H3","M1",
                          "mGluR5","MOR","NET","NMDA","VAChT","A2"]

        noradrenaline = ["NET", "A2"]

    receptors = {
        "source": source,
        "receptor_names": receptor_names,

        "serotonin": ["5HT1a","5HT1b","5HT2a","5HT4","5HT6","5HTT"],
        "acetylcholine": ["A4B2","M1","VAChT"],
        "noradrenaline": noradrenaline,
        "opioid": ["MOR"],
        "glutamate": ["mGluR5","NMDA"],
        "histamine": ["H3"],
        "gaba": ["GABAa"],
        "dopamine": ["D1","D2","DAT"],
        "cannabinnoid": ["CB1"],

        "exc": ['5HT2a','5HT4','5HT6','D1','mGluR5','A4B2','M1','NMDA'],
        "inh": ['5HT1a','5HT1b','CB1','D2','GABAa','H3','MOR','A2'],

        "receptor_label_formatted": [
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{1a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{1b}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{2a}}$', '$5\\text{-}\\mathrm{HT}_{\\mathrm{4}}$',
                '$5\\text{-}\\mathrm{HT}_{\\mathrm{6}}$', '$5\\text{-}\\mathrm{HTT}$',
                '$\\mathrm{A}_{\\mathrm{4}}\\mathrm{B}_{\\mathrm{2}}$', '$\\mathrm{M}_{\\mathrm{1}}$',
                '$\\mathrm{VAChT}$', '$\\mathrm{NET}$', '$\\mathrm{A}_{\\mathrm{2}}$',
                '$\\mathrm{MOR}$', '$\\mathrm{mGluR}_{\\mathrm{5}}$', '$\\mathrm{NMDA}$',
                '$\\mathrm{H}_{\\mathrm{3}}$', '$\\mathrm{GABA}_{\\mathrm{a}}$', '$\\mathrm{D}_{\\mathrm{1}}$',
                '$\\mathrm{D}_{\\mathrm{2}}$', '$\\mathrm{DAT}$', '$\\mathrm{CB}_{\\mathrm{1}}$'
            ]  ,
    }

    GROUP_KEYS = [
        "serotonin","acetylcholine","noradrenaline","opioid",
        "glutamate","histamine","gaba","dopamine","cannabinnoid"
    ]

    receptors["group_names"] = GROUP_KEYS
    receptors["receptor_groups"] = [receptors[k] for k in GROUP_KEYS]
    receptors["receptor_class"] = [receptors["exc"], receptors["inh"]]

    return receptors
