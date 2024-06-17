import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import pickle
from math import log
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import main_funcs as mf
from params_and_paths import *
from dominance_stats import dominance_stats

#to run code in parallel
num_workers = 8  # Set an appropriate number of workers

fmri_dir = mf.get_fmri_dir(DB_NAME)
subjects = mf.get_subjects(DB_NAME, fmri_dir)
subjects = [subj for subj in subjects if subj not in ignore[DB_NAME]]

output_dir = os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME,'first_level','regressions')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
y_names = np.array(['surprise','confidence', 'predictability', 'predictions'])
X_data = zscore(np.load(os.path.join(home_dir[DATA_ACCESS],'receptors', f'receptor_density_{MASK_NAME}.pickle'), allow_pickle=True))

#sklearn regression
columns = np.concatenate((receptor_names, np.array(["R2", "adjusted_R2", "BIC"])))

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

# for y_name in y_names:
#     results_df = pd.DataFrame(columns=columns)

#     for sub in subjects: 

#         y_data = zscore(np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten())
#         non_nan_indices = ~np.isnan(y_data)
#         X = X_data[non_nan_indices,:]
#         y = (y_data[non_nan_indices])

#         lin_reg = LinearRegression()
#         lin_reg.fit(X, y)
#         yhat = lin_reg.predict(X)
#         coefs = lin_reg.coef_

#         #adjusted R2
#         SS_Residual = sum((y - yhat) ** 2)
#         SS_Total = sum((y - np.mean(y)) ** 2)
#         r_squared = 1 - (float(SS_Residual)) / SS_Total
#         adjusted_r_squared = 1 - (1 - r_squared) * \
#             (len(y) - 1) / (len(y) - X.shape[1] - 1)
        
#         #BIC
#         num_params = len(lin_reg.coef_) + 1
#         mse = mean_squared_error(y, yhat)
#         bic = calculate_bic(len(y), mse, num_params)

#         #results by functional activity across participants 
#         results = pd.DataFrame([np.append(coefs, [r_squared, adjusted_r_squared, bic])], columns = results_df.columns)
#         results_df = pd.concat([results_df,results], ignore_index=True)

#     fname = f'{y_name}_{MASK_NAME}_regression_results_bysubject_all.csv'
#     results_df.to_csv(os.path.join(output_dir, fname), index=False)  


# #temporary: sanity check with dominance package
# from netneurotools import stats
# for y_name in y_names:
#     results_df = pd.DataFrame(columns=[receptor_names])
#     for sub in subjects:
#         print(f"--- dominance analysis SC subject {sub} ----")
#         y_data = np.load(os.path.join(home_dir[DATA_ACCESS],DB_NAME,MASK_NAME,'first_level',f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten()
#         non_nan_indices = ~np.isnan(y_data)
#         X = X_data[non_nan_indices,:]
#         y = y_data[non_nan_indices]
#         m, _ = stats.get_dominance_stats(X,y)
#         total_dominance_array = m["total_dominance"]
#         results = pd.DataFrame([total_dominance_array], columns=receptor_names)
#         results_df = pd.concat([results_df, results], ignore_index=True)
#     #save data 
#     output_dir = os.path.join(output_dir, 'temp')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     np.save(os.path.join(output_dir, f'{y_name}_{MASK_NAME}_dominance_bysubject_SC.pickle'), results_df) 


def process_subject(sub, y_name):
    print(f"--- dominance analysis for subject {sub} ----")
    y_data = zscore(np.load(os.path.join(home_dir[DATA_ACCESS], DB_NAME, MASK_NAME, 'first_level', f'sub-{sub:02d}_{y_name}_{MASK_NAME}_effect_size_map.pickle'), allow_pickle=True).flatten())
    non_nan_indices = ~np.isnan(y_data)
    X = X_data[non_nan_indices, :]
    y = y_data[non_nan_indices]
    m = dominance_stats(X, y)
    with open(os.path.join(output_dir, f'{y_name}_{MASK_NAME}_dominance_sub-{sub:02d}.pickle'), 'wb') as f:
        pickle.dump(m, f)
    total_dominance_array = m["total_dominance"]
    results = pd.DataFrame([total_dominance_array], columns=receptor_names)
    return results

for y_name in y_names:
    results_df = pd.DataFrame(columns=receptor_names)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_subject, sub, y_name) for sub in subjects]
        for future in futures:
            results = future.result()
            results_df = pd.concat([results_df, results], ignore_index=True)

    # Save data
    results_df.to_pickle(os.path.join(output_dir, f'{y_name}_{MASK_NAME}_dominance_allsubj.pickle'))
