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
from neuromaps import nulls, transforms, images 
from nilearn import datasets, surface
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018


paths = Paths()
params = Params()
rec = Receptors()

EXPLORE_MODEL = 'noEntropy_noER'

output_dir = os.path.join(paths.home_dir, 'variance_explained')
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

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_data =np.load(os.path.join(receptor_dir,f'receptor_density_schaefer_100.pickle'), allow_pickle=True)

with open(os.path.join(output_dir,'regression_null_models_grouplevel_schaefer100_vario.txt'), "w") as outfile:
    for task in tasks:
        outfile.write(f'{task}:\n')
        if task == 'Explore':
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level', EXPLORE_MODEL)
        else:
            task_path = os.path.join(paths.home_dir, task, 'schaefer', 'second_level')
                    
        if task in ['NAConf']:
            add_info = '_firstTrialsRemoved'
        elif not params.zscore_per_session:
            add_info = '_zscoreAll'
        else:
            add_info = ""

        for contrast in ['confidence', 'surprise']:
            print(f"--- variogram for {task} and {contrast} ----")

            fname = f'{contrast}_schaefer_effect_map{add_info}.nii.gz'
            data_vol = nib.load(os.path.join(task_path, fname))
            
            #masker
            atlas = fetch_atlas_schaefer_2018(n_rois=int(params.mask_details), resolution_mm=2) 
            atlas.labels = np.insert(atlas.labels, 0, "Background")
            masker = NiftiLabelsMasker(labels_img=atlas.maps) #parcelate
            data_parcel = masker.fit_transform(data_vol)
            map  = nib.load(atlas.maps)

            #get a parcelation gifti
            # labels = [label.decode() for label in atlas.labels]
            # parc_left = images.construct_shape_nii(atlas.map, labels=labels,
            #                                     intent='NIFTI_INTENT_LABEL')

            perms = nulls.burt2020(data_parcel, atlas='MNI152', density='2mm', n_perm=n_perm, parcellation=map)

            null_list = []
            for p in range(n_perm):
                null_img = image.new_img_like(data_parcel, perms[:,:,:,p])
                null_array = masker.fit_transform(null_img)
                null_list.append(null_array)
            null_arrays = np.concatenate(null_list, axis=0).T 

            emp, model_pval = get_reg_r_pval(zscore(receptor_data),
                        zscore(data_parcel.reshape(-1,1), axis=None), 
                        zscore(null_arrays), n_perm)
            
            outfile.write(f'{contrast}:\n')
            outfile.write(f'R2:{emp}  p:{model_pval}\n')
        outfile.write('\n\n')