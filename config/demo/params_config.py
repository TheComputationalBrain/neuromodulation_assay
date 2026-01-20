def load_params(task, cv_true=False):
    params = {
        'tasks': ["EncodeProb"],  #name of the task 
        'latent_vars': ["confidence", "surprise"], #name of the variable(s) to evaluate, the code will loop through these
        'ignore': [1, 4, 12, 20] #subjects that need to be ignored from the analysis
        }
    return params