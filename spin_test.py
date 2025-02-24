
import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from neuromaps import nulls, transforms
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from sklearn.metrics import r2_score
from scipy.stats import zscore
import main_funcs as mf
import glob
import pickle
import pandas as pd
import nibabel as nib


paths = Paths()
params = Params()
rec = Receptors()

n_spins = 2
EXPLORE_MODEL = 'noEntropy_noER'
tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']  

output_dir = os.path.join(paths.home_dir, 'variance_explained')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)


fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data')}

ignore = {'NAConf': [3, 5, 6, 9, 36, 51],
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': []}

cv_dir = os.path.join(paths.home_dir, 'variance_explained')
emp_df = pd.read_csv(os.path.join(cv_dir, 'overview_regression_cv_on_surf.csv'),index_col=0)
#get receptor data
receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_data =np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True)

#get index for receptor spin
index_array = np.arange(np.shape((receptor_data))[0])
spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k',
                                    n_perm=n_spins)

latent_vars = ['surprise', 'confidence'] 
df_pvalue = pd.DataFrame(index=tasks, columns=latent_vars)
df_r = pd.DataFrame(index=tasks, columns=latent_vars)

null_dist = []
all_null = []
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
                                
        mask_comb = params.mask 
        text = 'by voxel'

    for latent_var in latent_vars:
        emp = emp_df.loc[task, latent_var] #get empirical cv r2

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
            fmri_activity.append(effect_array) #? which dim?

        for s in range(spins.shape[1]):
            receptor_spin = receptor_data[spins[:, s], :] #permute receptors 

            r2_scores = [] 

            for i in range(len(subjects)):
                X_train = []
                y_train = []

                for j in range(len(subjects)):
                    if j != i:
                        # Remove NaNs
                        mask = ~np.isnan(fmri_activity[j]).flatten()
                        X_train.append(receptor_spin[mask])
                        y_train.append(zscore(fmri_activity[j].flatten()[mask]))

                # Concatenate training data
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)
                
                # Prepare test data for the left-out subject
                mask_test = ~np.isnan(fmri_activity[i]).flatten()
                X_test = receptor_spin[mask_test]
                y_test = zscore(fmri_activity[i].flatten()[mask_test])
                
                # Fit the model on the combined data
                model = LinearRegression(n_jobs=4)
                model.fit(X_train, y_train)
                
                # Predict on the left-out subject
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            average_r2 = np.mean(r2_scores)
            null_dist.append(average_r2)

            #save all R2 scores for each null: should result in a (subject, spin) matrix
            all_null.append(r2_scores)

        with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_null_cv_r2'), "wb") as fp:   
            pickle.dump(all_null, fp)
        null_r2 = np.mean(average_r2)

        #get p value 
        p_value = (1 + sum(null_dist > emp))/(spins.shape[1] + 1)
        #save the empiral value, mean spin r2 and p value to df 
        df_pvalue.loc[task, latent_var] = p_value
        df_r.loc[task, latent_var] = null_r2

df_pvalue.to_csv(os.path.join(output_dir,'overview_regression_null_cv_pvalues.csv'))
df_r.to_csv(os.path.join(output_dir,'overview_regression_null_cv_r2.csv'))
