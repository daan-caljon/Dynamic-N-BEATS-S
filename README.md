# Using dynamic loss weighting to boost improvements in forecast stability
This repository provides the code for the paper *"Using dynamic loss weighting to boost improvements in forecast stability"*.

The structure of the code is as follows:
```
Dynaminic-N-BEATS-S/
|_ R_scripts/
  |_ M4DataSet/                      #Download files from Kaggle
    |_ Monthly-test.csv               
    |_ Monthly-train.csv
  |_ M3-M4_evaluation.R              #Generate results + MCB plots
  |_ M3_statistical_baselines.R      #ETS, THETA, and ARIMA methods M3
  |_ M4_statistical_baselines.R      #"" M4
|_ data/
  |_ M3_monthly_TSTS.csv             #M3 monthly dataset
|_ scripts/
  |_ main.py                         #Script to train a dynamic N-BEATS-S model                 
|_ src/
  |_ data/
    |_ Read_data.py                  # Code to read in datasets
  |_ methods/
    |_ utils/        
      |_ metrics.py
    |_ Learner.py                    # Training methodology with dynamic loss weighting
    |_ NBEATSS.py                    # N-BEATS-S model
```

## Installation
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.10.13```.

## Usage
Change the ```DIR``` variable to your directory in ```main.py```. Set the relevant hyperparameters for the dynamic loss weighting (DLW) extension you want to run (see hyperparameter table in paper). Each run/sweep produces a ```.csv``` file with the forecasts.  [Weights & Biases](https://wandb.ai/site) is used to log (intermediary) results.

To run the statistical baselines, first download the M4 Monthly train and test ```.csv``` files from [Kaggle](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset). Save these files as indicated in the folder structure above. Then, run the ```*_statistical_baselines.R``` files.

To generate the results reported in the paper, run the different DLW extensions with 5 seeds. Next, put the ```.csv``` files into the right folders (see ```M3-M4_evaluation.R```). Running this file will generate both the tables and MCB plots.

## Acknowledgements
Our code builds upon the code from [Van Belle et al. (2023)](https://github.com/VerbekeLab/n-beats-s).

To implement the dynamic loss weighting methods we used the following repos as a starting point:

- Lucas Bo Tang for GradNorm: [https://github.com/LucasBoTang/GradNorm](https://github.com/LucasBoTang/GradNorm)
- Aviv Navon for Gradient Cosine Similarity Aviv Navon ([https://avivnavon.github.io/AuxiLearn/](https://avivnavon.github.io/AuxiLearn/))
- The LibMTL library for uncertainty weighting ([https://github.com/median-research-group/LibMTL](https://github.com/median-research-group/LibMTL))

Reference:

Van Belle, J., Crevits, R., & Verbeke, W. (2023). Improving forecast stability using deep learning. International Journal of Forecasting, 39(3), 1333-1350.
