import os
import glob
import numpy as np
import pandas as pd
import pickle
from math import log
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import zscore
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors

FROM_BETA = False
FROM_RECEPTOR = True
FROM_OLS = False

paths = Paths()
params = Params()
rec = Receptors()

fmri_dir = mf.get_fmri_dir(params.db)
subjects = mf.get_subjects(params.db, fmri_dir)
subjects = [subj for subj in subjects if subj not in params.ignore]

output_dir = os.path.join(paths.home_dir, 'variance_explained')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

if FROM_OLS:
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level', 'OLS')
else: 
    beta_dir  = os.path.join(paths.home_dir,params.db,params.mask,'first_level')
                                    
if params.parcelated:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
    mask_comb = params.mask + '_' + params.mask_details 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
    text = 'by region'
else:
    receptor_dir = os.path.join(paths.home_dir, 'receptors', 'PET') #vertex level analyis can only be run on PET data densities 
    receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
    mask_comb = params.mask 
    text = 'by voxel'

if rec.source == 'autorad_zilles44':
    #autoradiography dataset is only one hemisphere 
    receptor_density = np.concatenate((receptor_density, receptor_density))

if FROM_BETA:
    with open(os.path.join(output_dir,'predict_from_beta.txt'), "a") as outfile:
        outfile.write(f'{params.db}: variance explained in {text} analysis:\n\n')
        for latent_var in params.latent_vars:
            all_data = []
            all_rsquared = []
            for sub in subjects:
                    sub_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size.pickle'), allow_pickle=True).flatten()
                    all_data.append(sub_data)

            for i in range(len(all_data)): 
                sub_data = all_data[i]
                other_data = [arr for j, arr in enumerate(all_data) if j != i]
                other_data = np.stack(other_data)
                mean_data = np.nanmean(other_data, axis=0)

                non_nan_indices = ~np.isnan(sub_data)
                X = mean_data[non_nan_indices].reshape(-1, 1) #non parcelated data might contain a few NaNs in perihery from voxels with constant activation 
                y = sub_data[non_nan_indices]

                lin_reg = LinearRegression()
                lin_reg.fit(X, y)
                rsquared = lin_reg.score(X, y)
                all_rsquared.append(rsquared)

            expl_variance = np.mean(all_rsquared)
            outfile.write(f'{latent_var}: {expl_variance}\n')

        outfile.write('\n\n\n\n')


if FROM_RECEPTOR:
    with open(os.path.join(output_dir,'predict_from_receptor.txt'), "a") as outfile:
        outfile.write(f'{params.db}: variance explained in {text} analysis with {rec.source} as predictor:\n\n')
        n_features = receptor_density.shape[1]
        for latent_var in params.latent_vars:
            r2_scores = [] 
            fmri_files = glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size.pickle'))
            fmri_activity = []
            for file in fmri_files:
                with open(file, 'rb') as f:
                    fmri_activity.append(pickle.load(f))

            for i in range(len(subjects)):
                X_train = []
                y_train = []

                for j in range(len(subjects)):
                    if j != i:
                        # Remove NaNs
                        if params.parcelated:
                            non_nan_region = ~np.isnan(receptor_density).any(axis=1)
                            mask = np.where(non_nan_region)[0]
                            X_train.append(receptor_density[mask])
                            y_train.append(zscore(fmri_activity[j].flatten()[mask]))
                        else:
                            mask = ~np.isnan(fmri_activity[j]).flatten()
                            X_train.append(receptor_density[mask])
                            y_train.append(zscore(fmri_activity[j].flatten()[mask]))
            
                # Concatenate training data
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)
                
                # Prepare test data for the left-out subject
                if params.parcelated:
                    non_nan_region = ~np.isnan(receptor_density).any(axis=1)
                    mask_test = np.where(non_nan_region)[0]
                    X_test = receptor_density[mask]
                    y_test = zscore(fmri_activity[i].flatten()[mask])
                else:
                    mask_test = ~np.isnan(fmri_activity[i]).flatten()
                    X_test = receptor_density[mask_test]
                    y_test = zscore(fmri_activity[i].flatten()[mask_test])
                
                # Fit the model on the combined data
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predict on the left-out subject
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            average_r2 = np.mean(r2_scores)
            outfile.write(f'{latent_var}: {average_r2}\n')

        outfile.write('\n\n\n\n')

    


