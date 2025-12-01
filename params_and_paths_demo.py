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





