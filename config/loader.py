from config.manuscript.params_config import load_params
from config.manuscript.paths_config import load_paths
from config.manuscript.receptor_config import load_receptors
from config.utils import AttrDict


def load_config(task: str, source="PET2", cv=False, return_what="all"):
    """
    Load configuration for a task.
    
    return_what: 
        "all"         -> return (params, paths, receptors)
        "params"      -> return only params
        "paths"       -> return only paths
        "receptors"   -> return only receptors
    """

    params = AttrDict.convert(load_params(task, cv_true=cv))
    paths = AttrDict.convert(load_paths(task))
    receptors = AttrDict.convert(load_receptors(source))

    if return_what == "params":
        return params
    elif return_what == "paths":
        return paths
    elif return_what == "receptors":
        return receptors

    # default = return all three
    return params, paths, receptors
