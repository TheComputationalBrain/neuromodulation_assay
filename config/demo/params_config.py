def load_params(task, cv_true=False):
    params = {
        'tasks': ['Study_1'],  #name of the task 
        'latent_vars': ["Confidence"], #name of the variable to evaluate 
        'ignore': [1, 4, 12, 20] #subjects that need to be ignored from the analysis
        }
    return params