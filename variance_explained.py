import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
from neuromaps import transforms
from math import log
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import zscore, sem, ttest_rel
import main_funcs as mf
from params_and_paths import Paths, Params, Receptors

FROM_BETA = False
FROM_RECEPTOR = False
ON_SURFACE = True
COMP_NULL = True

if ON_SURFACE:
    proj = '_on_surf'
else:
    proj = ''

paths = Paths()
params = Params()
rec = Receptors()

output_dir = os.path.join(paths.home_dir, 'variance_explained')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 

fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data')}

ignore = {'NAConf': [3, 5, 6, 9, 36, 51],
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': []}

if FROM_BETA:
    for task in tasks: 

        if task == 'Explore':
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
        else:
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        subjects =  mf.get_subjects(task, fmri_dir[task])
        subjects = [subj for subj in subjects if subj not in ignore[task]] 
                                            
        if params.parcelated:
            mask_comb = params.mask + '_' + params.mask_details 
            text = 'by region'
        else:
            mask_comb = params.mask 
            text = 'by voxel'

        with open(os.path.join(output_dir,f'predict_from_beta{proj}.txt'), "a") as outfile:
            outfile.write(f'{task}: variance explained in analysis:\n\n')
            for latent_var in ['surprise', 'confidence']:
                all_data = []
                all_rsquared = []
                for sub in subjects:
                        if ON_SURFACE:
                            file = nib.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size_map{add_info}.nii.gz'))
                            effect_data = transforms.mni152_to_fsaverage(file, fsavg_density='41k')
                            data_gii = []
                            for img in effect_data:
                                data_hemi = img.agg_data()
                                data_hemi = np.asarray(data_hemi).T
                                data_gii += [data_hemi]
                                effect_array = np.hstack(data_gii)    
                            all_data.append(effect_array) 
                        else:
                            sub_data = np.load(os.path.join(beta_dir,f'sub-{sub:02d}_{latent_var}_{mask_comb}_effect_size{add_info}.pickle'), allow_pickle=True).flatten()
                            all_data.append(sub_data)

                for i in range(len(all_data)): 
                    sub_data = all_data[i]
                    other_data = [arr for j, arr in enumerate(all_data) if j != i]
                    other_data = np.stack(other_data)
                    mean_data = np.nanmean(other_data, axis=0)

                    if ON_SURFACE:
                        mask = ~np.logical_or(np.isnan(sub_data), np.isclose(sub_data,0)).flatten() 
                    else:
                        mask = ~np.isnan(sub_data)
                    X = mean_data[mask].reshape(-1, 1) #non parcelated data might contain a few NaNs in perihery from voxels with constant activation 
                    y = sub_data[mask]

                    lin_reg = LinearRegression()
                    lin_reg.fit(X, y)
                    rsquared = lin_reg.score(X, y)
                    all_rsquared.append(rsquared)

                expl_variance = np.mean(all_rsquared)
                sem_r2 = sem(all_rsquared)
                outfile.write(f'{latent_var}: {expl_variance}, sem: {sem_r2}\n')
                #save empirical results as pd

            outfile.write('\n\n')

if FROM_RECEPTOR:
    latent_vars = ['surprise', 'confidence'] 
    df = pd.DataFrame(index=tasks, columns=latent_vars)

    for task in tasks: 

        if task == 'Explore':
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
        else:
            beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

        subjects =  mf.get_subjects(task, fmri_dir[task])
        subjects = [subj for subj in subjects if subj not in ignore[task]] 

        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""    

        if ON_SURFACE: 
            receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
            receptor_density =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_on_surf.pickle'), allow_pickle=True))
            mask_comb = params.mask 
        else:                                            
            if params.parcelated:
                receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
                mask_comb = params.mask + '_' + params.mask_details 
                receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
                text = 'by region'
            else:
                receptor_dir = os.path.join(paths.home_dir, 'receptors', 'PET2') #vertex level analyis can only be run on PET data densities 
                receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
                mask_comb = params.mask 
                text = 'by voxel'

            if rec.source == 'autorad_zilles44':
                #autoradiography dataset is only one hemisphere 
                receptor_density = np.concatenate((receptor_density, receptor_density))

        with open(os.path.join(output_dir,f'predict_from_receptor{proj}.txt'), "a") as outfile:
            outfile.write(f'{task}: variance explained in analysis with {rec.source} as predictor:\n\n')
            n_features = receptor_density.shape[1]
            for latent_var in latent_vars:
                r2_scores = [] 
                if ON_SURFACE:
                    fmri_files = glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size_map{add_info}.nii.gz'))
                    fmri_activity = []
                    for file in fmri_files:
                        data_vol = nib.load(file)
                        effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
                        data_gii = []
                        for img in effect_data:
                            data_hemi = img.agg_data()
                            data_hemi = np.asarray(data_hemi).T
                            data_gii += [data_hemi]
                            effect_array = np.hstack(data_gii)    
                        fmri_activity.append(effect_array) 
                else:    
                    fmri_files = glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size{add_info}.pickle'))
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
                                mask = np.logical_or(np.isnan(fmri_activity[j]), np.isclose(fmri_activity[j],0)).flatten()
                                X_train.append(receptor_density[~mask])
                                y_train.append(zscore(fmri_activity[j].flatten()[~mask]))
                
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
                        mask_test = np.logical_or(np.isnan(fmri_activity[i]), np.isclose(fmri_activity[i],0)).flatten()
                        X_test = receptor_density[~mask_test]
                        y_test = zscore(fmri_activity[i].flatten()[~mask_test])
                    
                    # Fit the model on the combined data
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Predict on the left-out subject
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    r2_scores.append(r2)

                average_r2 = np.mean(r2_scores)
                sem_r2 = sem(r2_scores)
                outfile.write(f'{latent_var}: {average_r2}, sem: {sem_r2}\n')
                df.loc[task, latent_var] = average_r2

                with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2{proj}.pickle'), "wb") as fp:   
                    pickle.dump(r2_scores, fp)

            outfile.write('\n\n')
    df.to_csv(os.path.join(output_dir,f'overview_regression_cv{proj}.csv'))


if COMP_NULL:
    latent_vars = ['surprise', 'confidence'] 
    df = pd.DataFrame(index=tasks, columns=latent_vars)

    with open(os.path.join(output_dir,f'compare_emp_null_cv_{proj}.txt'), "w") as outfile:
        for task in tasks: 
            for latent_var in latent_vars:
                outfile.write(f'{task} and {latent_var}:\n')
                #get the empirical r2 (on surf)
                emp = np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2_on_surf.pickle'), allow_pickle=True)
                #get the null r2
                null = np.load(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_null_cv_r2.pickle'), allow_pickle=True)
                null_df = pd.DataFrame(null)
                null_means = null_df.mean().tolist()

                #summarize the null r2 as we did with the empirical r2 above (for table overview in paper)
                r2 = sum(null_means) / len(null_means)
                df.loc[task, latent_var] = r2

                #compare the left-out subject (mean) r2 via t-test 
                ttest = ttest_rel(null_means, emp, alternative='less') 
                outfile.write(f't-value: {ttest.statistic}\n') 
                outfile.write(f'p-value: {ttest.pvalue}\n')
                outfile.write(f'df: {ttest.pvalue}\n')
                outfile.write(f'\n\n')

            df.to_csv(os.path.join(output_dir,f'overview_null_cv{proj}.csv'))






