# neuromodulation_assay

This repository contains code for an investigation into the associations between neurotransmitter systems and computational variables of learning under uncertainty. 

## params

`DB_NAME` is the name of the dataset to be (re-)analized
- we are currently reanalizing 4 datasets from our group: 3 probability learning tasks and one reward learning task
- the datasets differ in their filestructure, however, the analysis is the same across datasets

`MASK_NAME` is the name of the mask used
- we currently only analyze cortical data

`PARCELATED` determines whether the analysis should return by-region results. If set to true, a corresponding atlas as to be assigned to MASK_NAME

`RECEPTOR_SOURCE` determines which dataset is used to as an estimate of receptor density
- PET2 is the PET dataset published in Hansen, J. Y., Shafiei, G., Markello, R. D., Smart, K., Cox, S. M., Nørgaard, M., ... & Misic, B. (2022). Mapping neurotransmitter systems to the structural and functional organization of the human neocortex. Nature neuroscience, 25(11), 1569-1581. (see also the corresponding [github](https://github.com/netneurolab/hansen_receptors)) and an α2 map provided by the authors of Laurencin, C., Lancelot, S., Merida, I., Costes, N., Redouté, J., Le Bars, D., ... & Ballanger, B. (2023). Distribution of α2-adrenergic receptors in the living human brain using [11C] yohimbine PET. Biomolecules, 13(5), 843. This is the dataset we ended up using in the final poster.
- autorad_zilles44 cortical receptor densities originally published in Zilles & Palomero-Gallagher (2017), made available here: https://github.com/AlGoulas/receptor_principles
- AHBA is the Allen human brain atlas microarray dataset

`UPDATE_REG` can be set to true to use update as a computational latent variable of interest. Otherwise suprise and confidence are used.

## code

- [receptor_density_PET.py](receptor_density_PET.py) calculates a weighted average of PET maps that used the same tracer and returns a voxel-by-receptor matrix. It also plots the receptor densities and their correlations.
- [run_glm.py](run_glm.py) creates the design matrix, runs the glm and save the contrasts
- [receptor_regression.py](receptor_regression.py) runs a multiple regression and dominance analysis with recptor density as a predictor
- [plot_reg.py](plot_reg.py) plots the final domiance plots found on the poster as well as the full model regression weigths

