import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from dominance_analysis import Dominance
import main_funcs as mf
from params_and_paths import *

fmri_dir = mf.get_fmri_dir(DB_NAME)
subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

output_dir = os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME,'first_level','regressions')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
receptor_names_core = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                            "D1", "D2", "DAT", "GABAa", "M1", "mGluR5",
                            "NET", "NMDA", "VAChT"])

receptor_set = set(receptor_names)
core_set = set(receptor_names_core)
columns_to_drop = list(receptor_set - core_set)

receptors_to_remove = [7, 14, 16] #opioid, histamin, cannabinoid
y_names = np.array(['surprise','confidence', 'predictability', 'predictions'])
X_data = np.load(os.path.join(home_dir[DATA_ACCESS],'receptors', f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True)
X_data_core = np.delete(X_data, receptors_to_remove, axis=1)

#sklearn regression
columns = np.concatenate((receptor_names_core, np.array(["R2", "adjusted_R2"])))

for y_name in y_names:
    results_df = pd.DataFrame(columns=columns)

    for sub in subjects: 

        y_data = np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten()
        non_nan_indices = ~np.isnan(y_data)
        X = X_data_core[non_nan_indices,:]
        y = y_data[non_nan_indices]

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        yhat = lin_reg.predict(X)
        coefs = lin_reg.coef_

        #adjusted R2
        SS_Residual = sum((y - yhat) ** 2)
        SS_Total = sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (float(SS_Residual)) / SS_Total
        adjusted_r_squared = 1 - (1 - r_squared) * \
            (len(y) - 1) / (len(y) - X.shape[1] - 1)
        
        #results by functional activity across participants 
        results = pd.DataFrame([np.append(coefs, [r_squared, adjusted_r_squared])], columns = results_df.columns)
        results_df = pd.concat([results_df,results], ignore_index=True)

    fname = f'{y_name}_regression_results_bysubject.csv'
    results_df.to_csv(os.path.join(output_dir, fname), index=False)  


#dominance analysis
#package expects a dataframe
def create_dataframe(y_array, y_name, X, receptor_names):
    data = {y_name: y_array}
    df = pd.DataFrame(data)
    df_2d = pd.DataFrame(X, columns=receptor_names)
    result_df = pd.concat([df, df_2d], axis=1)
    return result_df.drop(columns=columns_to_drop)

#for name in y_names:
for y_name in y_names:
    results_df = pd.DataFrame(columns=[receptor_names_core])
    for sub in subjects:
        y_data = np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten()
        data_reg = create_dataframe(y_data, y_name, X, receptor_names) 
        data_reg = data_reg.dropna()
        dominance_regression=Dominance(data=data_reg,target=y_name,top_k=19) #TODO find out how to access the output 
        incr_variable_rsquare=dominance_regression.incremental_rsquare()
        dominance_stats = dominance_regression.dominance_stats()
        results = pd.DataFrame([dominance_stats["Total Dominance"].values], columns=receptor_names_core)
        results_df = pd.concat([results_df, results], ignore_index=True)
    #save data 
    fname = f'{y_name}_dominance_results_bysubject.pickle'
    results_df.to_csv(os.path.join(output_dir, fname), index=False)  


#temporary: sanity check for dominance package:
from netneurotools import stats
for y_name in y_names:
    results_df = pd.DataFrame(columns=[receptor_names_core])
    for sub in subjects:
        y_data = np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten()
        non_nan_indices = ~np.isnan(y_data)
        X = X_data_core[non_nan_indices,:]
        y = y_data[non_nan_indices]
        m, _ = stats.get_dominance_stats(X,y)
        results = pd.DataFrame([m["total_dominance"].values], columns=receptor_names_core)
        results_df = pd.concat([results_df, results], ignore_index=True)
    #save data 
    output_dir = os.path.join(output_dir, 'temp')
    if not os.path.exists(fname):
        os.makedirs(fname)
    np.save(os.path.join(output_dir, f'{y_name}_dominance2_results_bysubject.pickle'), results_df) 