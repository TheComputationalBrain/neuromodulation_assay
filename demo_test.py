#load packages
import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from glob import glob
import pickle
import numpy as np
from nilearn import plotting, datasets, image
from nilearn import surface
from config.loader_demo import load_config
from utils.main_funcs import nii_to_cortical_voxel_array
from analysis.regression_cv_with_spintest import prepare_spins, run_reg_cv_with_spin
from analysis.variance_explained import run_comp_null, run_predict_from_beta, run_group_ratio_summary
from analysis.receptor_effect_map_relationship  import run_dominance_analysis, load_dominance_data, aggregate_dominance, plot_dominance_bars

params, paths, rec = load_config('study_1', return_what='all')

# spins = prepare_spins(paths, rec, n_spins=1000)

# run_reg_cv_with_spin(params, paths, rec, spins, output_dir=os.path.join(paths.results_dir, 'regressions'), run_spin = True)

##uncomment this to run the dominance analysis
run_dominance_analysis(
            params=params,
            paths=paths,
            rec=rec,
            model_type="linear",
            start_at=0,
            num_workers=30,
            output_dir=os.path.join(paths.results_dir, 'dominance') 
        )