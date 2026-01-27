# Neuromodulatory systems partially account for the topography of cortical networks of learning under uncertainty

This repository contains the code and data associated with the manuscript  
**“Neuromodulatory systems partially account for the topography of cortical networks of learning under uncertainty.”**

All code was written in **Python 3.10.12**. Versions of the used packages can be found in **`requirements.txt`**

To explore neuromodulation and receptor/transporter contributions in *user-provided* fMRI data, see the notebook **`analysis_demo.ipynb`**. This notebook demonstrates how to apply the analysis pipeline described in the manuscript to new fMRI effect maps.

Below, the repository structure is described in an order that mirrors the organization of the manuscript.

---

## Repository Structure

### `preprocessing/`

This folder contains all code specific to preprocessing and analyzing the probabilistic learning–related datasets used in the paper.

- **`receptor_density_PET.py`**  
  Processes PET receptor/transporter density maps from Hansen et al. (2020) (see original GitHub repository).  
  The resulting receptor/transporter dataframe is used in downstream analyses.

- **`run_glm.py`**  
  Fits a GLM to the fMRI data to obtain the effect maps analyzed in the manuscript.  
  Regressors are derived from an ideal observer model specified in `TransitionProbModel`.  
  Design matrix details are implemented in `functions_design_matrices.py`.

- **`second_level_analysis.py`**  
  Runs group-level analyses of fMRI effect maps.

---

### `behavior/`

**`behavior_meta_analysis.py`**  
Runs the behavioral analysis reported in the manuscript.
 *Corresponds to Figure 1 in the manuscript.*

---

### `analysis/`

This folder contains scripts for all main statistical analyses and figures. 
The original analysis scripts assume a specific folder structure and namining convention, do apply code to new fMRI effect maps please use the notebook provided which has simplified these dependencies. 

- **`effect_map_correlations.py`**  
  Performs spatial permutation (“spin”) tests to assess the significance of correlations between fMRI maps.

- **`invariance_effect_maps.py`**  
  Tests and visualizes the invariance of fMRI effect map topographies.  
  Statistical significance is determined using results from `effect_map_correlations.py`.  
  *Corresponds to Figure 2 in the manuscript.*

- **`regression_cv_with_spin_test.py`**  
  Runs cross-validated multiple regression models along with a spatial null model (spin test).

- **`variance_explained.py`**  
  Summarizes and compares explained variance from the regression analyses in `regression_cv_with_spin_test.py`.  
  *Corresponds to Figure 3 in the manuscript.*

- **`receptor_effect_map_relationship.py`**  
  Examines relationships between fMRI effect maps and individual receptor/transporter densities.  
  *Corresponds to Figure 4 in the manuscript.*

- **`plots_analysis_paper.py`**  
  High-level script that reproduces all figures reported in the manuscript.

---

### `config/`

- **`config/manuscript/`**  
  Contains configuration loaders specifying parameters, paths, and receptor-specific information for each task used in the manuscript.

- **`config/demo/`**  
  Simplified configuration loader for running receptor/transporter analyses on new datasets.  
  Modify this loader when applying the pipeline to user-provided data.

---

### `data/`

- **`data/beta_dir/`**  
  Contains fMRI effect maps for all studies included in the manuscript, organized by study (Study 1–Study 4).  
  To analyze a new dataset, add correctly named and preprocessed effect maps in a new subfolder.  
  See `analysis_demo.ipynb` for details.

- **`data/receptor_dir/`**  
  Receptor/transporter densities (voxel x receptor) in both volumetric and surface formats.

  For the original PET data please see the following publication and the associated [GitHub](https://github.com/netneurolab/hansen_receptors) 

  Hansen et al. (2022). Mapping neurotransmitter systems to the structural and functional organization of the human neocortex. Nature neuroscience, 25(11), 1569-1581.

  Note that in the manuscript we addtionaly used a a2 densitity map from:

  Laurencin et al. (2023). Distribution of α2-adrenergic receptors in the living human brain using [11C] yohimbine PET. Biomolecules, 13(5), 843.

  Please contact the authors directly to optain the map. 

---

### `results/`

Contains precomputed regression results, maximum variance explained, and dominance analysis outputs.  
All results can be reproduced by running the provided notebook.