# null model in volumes 
#NB: this is comutationally very intensive: group level analysis only for now 
import os 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import tempfile
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from neuromaps import nulls, transforms
from nilearn import datasets, surface
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from nilearn import image, datasets


paths = Paths()
params = Params()
rec = Receptors()

EXPLORE_MODEL = 'noEntropy_noER'

output_dir = os.path.join(paths.home_dir, 'domain_general')
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    return r_squared

def get_reg_r_pval(X, y, spins, nspins):
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X, spins[:, s])
    return emp, (1 + sum(null > emp))/(nspins + 1)

#sanity check: get the R2 group level from the volumne regression
template = datasets.load_mni152_template(resolution=2)
tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']  
n_perm = 1000

with open(os.path.join(output_dir,'regression_null_models_grouplevel.txt'), "w") as outfile:
    for task in tasks:
        outfile.write(f'{task}:\n')
        if task == 'Explore':
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
        else:
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')

        receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
        receptor_data =np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True)
                    
        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        for contrast in ['confidence']:
            fname = f'{contrast}_schaefer_effect_map{add_info}.nii.gz'
            data_vol = nib.load(os.path.join(task_path, fname))
            

            #masker
            masker = fun.get_masker()
            masker.fit()
            data_vol =image.resample_img(data_vol, template.affine)  #make sure to resample to mni 2mm
            #? masking returns a shorter array, is that a problem fot the null model?
            data_array = masker.fit_transform(data_vol)
            nulls = nulls.burt2020(data_vol, atlas='MNI152', density='2mm', n_perm=n_perm)

            null_list = []
            for p in range(n_perm):
                null_img = image.new_img_like(data_vol, nulls[:,:,:,p])
                null_array = masker.fit_transform(null_img)
                null_list.append(null_array)
            null_arrays = np.concatenate(null_list, axis=0).T 
            
            #do the nulls have zeros at the same spots?
            #adjust the receptor data so it has the same values 
            emp, model_pval = get_reg_r_pval(zscore(receptor_data),
                        zscore(data_array.reshape(-1,1), axis=None), 
                        zscore(null_arrays), n_perm)
            
            outfile.write(f'{contrast}:\n')
            outfile.write(f'R2:{emp}  p:{model_pval}\n')
        outfile.write('\n\n')