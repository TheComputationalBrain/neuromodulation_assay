
import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import zscore
import concurrent.futures
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from neuromaps import nulls, transforms
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import main_funcs as mf
import glob
import pickle
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr


paths = Paths()
params = Params()
rec = Receptors()

n_spins = 1000
MODEL_TYPE = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'
RUN_SPIN = True
EXPLORE_MODEL = 'noEntropy_noER'
FIX_PARTIAL_COVERAGE = True
SCORE = 'corr' # determination or corr

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 
latent_vars = ['surprise', 'confidence'] 

output_dir = os.path.join(paths.home_dir, 'variance_explained')
os.makedirs(output_dir, exist_ok=True)

fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data'),
            'lanA': '/home_local/alice_hodapp/language_localizer/'}

ignore = {'NAConf': [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59],
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': [], 
            'lanA' : [88]} 


receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_data =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
index_array = np.arange(receptor_data.shape[0])
spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k', n_perm=n_spins)
mask_comb = params.mask

def compute_loocv_r2_with_equal_mask(receptor_map_full, fmri_activity, common_mask, score='determination'):
    """
    receptor_map_full: (n_vertices, n_features)
    fmri_activity: list of (n_vertices,) arrays for each subject
    returns: list of per-fold r2s
    """
    n_subj = len(fmri_activity)
    r2_scores = []

    for i in range(n_subj):
        X_train_list, y_train_list = [], []
        for j in range(n_subj):
            if j == i: continue
            Xj = receptor_map_full[common_mask, :]
            yj = fmri_activity[j].flatten()[common_mask]
            yj = (yj - np.nanmean(yj)) / (np.nanstd(yj) + 1e-12)
            X_train_list.append(Xj)
            y_train_list.append(yj)

        X_train = np.concatenate([x for x in X_train_list], axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_test = receptor_map_full[common_mask, :]
        y_test = fmri_activity[i].flatten()[common_mask]
        y_test = zscore(y_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if score == 'determination':
            r2_scores.append(r2_score(y_test, y_pred))
        if score == 'corr':
            r, _ = pearsonr(y_test, y_pred)
            r2_scores.append(r**2)

    return r2_scores

def process_task(task):
    results_emp = {}
    results_null = {}

    if task == 'Explore':
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
    elif task == 'lanA':
        beta_dir = fmri_dir['lanA']
    else:
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')    
        
    subjects = [subj for subj in mf.get_subjects(task, fmri_dir[task]) if subj not in ignore[task]]

    add_info = '_firstTrialsRemoved' if task == 'NAConf' else ''
    
    for latent_var in latent_vars:
        if task == 'lanA':
            fmri_files = []
            for subj in subjects:
                subj_id = f"{subj:03d}"  
                pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
                fmri_files.extend(glob.glob(pattern))
        else:
            fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size_map{add_info}.nii.gz')))
            fmri_files = []
            for file in fmri_files_all:
                basename = os.path.basename(file)
                subj_str = basename.split('_')[0]  # 'sub-XX'
                subj_id = int(subj_str.split('-')[1])  # XX as integer
                if subj_id in subjects:
                    fmri_files.append(file)
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

        all_valid_masks = [
        ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
        for arr in fmri_activity
        ]
        common_mask = np.logical_and.reduce(all_valid_masks)
        invalid_mask = ~common_mask

        if FIX_PARTIAL_COVERAGE:
            receptor_neutral_z = receptor_data.copy().astype(float)
            for f in range(receptor_data.shape[1]):
                rep_value = np.nanmean(receptor_data[common_mask, f])
                receptor_neutral_z[invalid_mask, f] = rep_value
                receptor_neutral_z[:, f] = zscore(receptor_neutral_z[:, f]) #? zscore each receptor or over all
        else:
            receptor_neutral_z = receptor_data

        print(f"--- CV for {task} and {latent_var} ---")  
        r2_scores = compute_loocv_r2_with_equal_mask(receptor_neutral_z, fmri_activity, common_mask, score=SCORE)
        with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r_2{SUFFIX}.pickle'), "wb") as fp:   
            pickle.dump(r2_scores, fp)
        average_r2 = np.mean(r2_scores)
        results_emp[(task, latent_var)] = average_r2
        
        if RUN_SPIN and MODEL_TYPE == 'linear':
            print(f"--- Spin test for {task} and {latent_var} ---")
            all_null = []
            for s in range(spins.shape[1]):
                receptor_spin = receptor_neutral_z[spins[:, s], :]
                r2_scores = compute_loocv_r2_with_equal_mask(receptor_spin, fmri_activity, common_mask, score=SCORE)
                all_null.append(r2_scores)
            with open(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2{SUFFIX}.pickle'), "wb") as fp:
                pickle.dump(all_null, fp)
            all_null_means = [np.mean(r2) for r2 in all_null]
            average_r2 = np.mean(all_null_means)
            results_null[(task, latent_var)] = average_r2
        elif RUN_SPIN and MODEL_TYPE != 'linear':
            raise NotImplementedError("Spin test not implemented for non-linear models.")
        else:
            pass

    return results_emp, results_null


# Main
all_emp, all_null = {}, {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_task, task) for task in tasks]
    for f in futures:
        emp, null = f.result()
        all_emp.update(emp)
        all_null.update(null)

# Convert to DataFrames
df_emp = pd.DataFrame.from_dict(all_emp, orient="index", columns=["r2"])
df_emp.index = pd.MultiIndex.from_tuples(df_emp.index, names=["task", "latent_var"])
df_emp.to_csv(os.path.join(output_dir, f"overview_regression_cv_{SCORE}.csv"))

df_null = pd.DataFrame.from_dict(all_null, orient="index", columns=["r2"])
df_null.index = pd.MultiIndex.from_tuples(df_null.index, names=["task", "latent_var"])
df_null.to_csv(os.path.join(output_dir, f"overview_null_cv_{SCORE}.csv"))

