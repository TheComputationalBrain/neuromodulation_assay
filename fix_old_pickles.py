"""
Created on Wed Apr 09 11:01:15 2025

This scripts sets the parameters so that old pickles can still be loaded and then saves them as csv
Script works with pandas 2.2.2 and Numpy 1.26.4, might need to be adapted for other versions

@author: Alice
"""
import types
import sys
import os
import pickle
import pandas as pd
import numpy as np 

# --- Patch missing pandas index types for old pickles ---
fake_numeric = types.ModuleType("pandas.core.indexes.numeric")
fake_numeric.Int64Index = type(pd.Index([1, 2, 3]))
fake_numeric.RangeIndex = type(pd.RangeIndex(3))
sys.modules['pandas.core.indexes.numeric'] = fake_numeric

#set paths
PICKLE_DIR = '/neurospin/unicog/protocols/comportement/ConfidenceDataBase_2020_Meyniel/subjects_data'
list_pickles = [f for f in os.listdir(PICKLE_DIR) if f.endswith(".pickle")]

for file in list_pickles:
    with open(os.path.join(PICKLE_DIR, file), 'rb') as file_info:
        pickle_file = pickle.load(file_info, encoding='bytes')

    if isinstance(pickle_file, pd.DataFrame):
        pickle_file.to_csv(os.path.join(PICKLE_DIR, file.replace(".pickle", ".csv")), index=False)
    
    elif isinstance(pickle_file, (list, np.ndarray)):
        # Convert to DataFrame first, then save as CSV
        df = pd.DataFrame(pickle_file)
        df.to_csv(os.path.join(PICKLE_DIR, file.replace(".pickle", ".csv")), index=False)
        
    else:
        print(f"Unsupported type for file: {file}, please add to code")