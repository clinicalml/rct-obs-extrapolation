# rct-obs-extrapolation
Code for paper, "Falsification before Extrapolation in Causal Effect Estimation"

Working directory for project on validating observational studies with a randomized clinical trial that has partial overlap for robust inference on the non-overlap region. **This repo is currently under construction. Check back later for end-to-end notebooks recreating the results of our NeurIPS 2022 Paper, "Falsification before Extrapolation in Causal Effect Estimation"**

## Datasets

* ihdpFull.csv: Full covariate dataset with 985 observations and 103 variables
* ihdp.csv: The dataset with covariates used in (Hill, 2011) and three additional variables indicating the mom's ethnicity, ending with 985 observations and 29 variables (including 1 treatment variable). Note that part of the observations was further removed in (Hill, 2011) to create confounding, which is not done to this dataset.

## Access to Women's Health Initiative Dataset
Request access to the WHI dataset used in this work by going to the following URL: https://biolincc.nhlbi.nih.gov/studies/whi_ctos/, and clicking "Request Data." 
