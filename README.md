# Using dynamic loss weighting to boost improvements in forecast stability
This repository provides the code for the paper *"Using dynamic loss weighting to boost improvements in forecast stability"*.

The structure of the code is as follows:
```
Dynamic-N-BEATS-S/
|_ data/
  |_ M3_monthly_TSTS.csv             #M3 monthly dataset
|_ Graph_data/                       #Data to generate the figures
|_ R_scripts/                        #R scripts + forecasts for each method
|_ scripts/
  |_ figures                         #Scripts to generate the figures in the paper 
  |_ M3                              #Scripts for M3 forecasts
  |_ M4                              #Scripts for M4 forecasts
  |_ Kappa_tuning_M3.py              #Scripts for M3 Figure 8 forecasts
  |_ Kappa_tuning_M4.py              #Scripts for M4 Figure 8 forecasts
  |_ main.py                         #Generic script to train a dynamic N-BEATS-S model 
  |_ run_all_methods_M3.py           #Run all scripts in M3 folder
  |_ run_all_methods_M4.py           #Run all scripts in M4 folder  
|_ src/
  |_ data/                           
    |_ Read_data.py                  # Code to read in datasets
  |_ methods/
    |_ utils/        
      |_ metrics.py
      |_ NashMTL.py
    |_ Learner.py                    # Training methodology with dynamic loss weighting
    |_ NBEATSS.py                    # N-BEATS-S model
```

## Installation
The ```requirements.txt``` provides the necessary packages. All code was written for ```python 3.10.13```. The exact environment that was used to run the experiments is given in ```environment.yaml```. The R scripts were developed in ```R version 4.3.2``` and use the following packages: 
- **Base packages**:  
  `stats`, `graphics`, `grDevices`, `utils`, `datasets`, `methods`, `base`

- **Attached packages**:  
  - `tsutils 0.9.4`  
  - `DescTools 0.99.54`  
  - `stringr 1.5.1`  
  - `Mcomp 2.8`  
  - `forecast 8.21.1`  
  - `magrittr 2.0.3`  
  - `openxlsxx 4.2.5.2`  
  - `data.table 1.15.0`
## Data
The M4 Monthly and M3 Monthly publicly available datasets are used. The M3 dataset is provided in the ```data``` folder. The M4 dataset is automatically downloaded online in ```Learner.py```. For the R scripts, the M4 dataset is provided as the ```Monthly-test.csv``` and ```Monthly-train.csv``` files. These files can be found on [Kaggle](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset).
## Usage
To efficiently run this code, access to a CUDA-enabled GPU is required.

Change the ```wandb_project_name``` to your ```wandb``` project in the script you want to run. Each run/sweep produces a ```.csv``` file with the forecasts. The scripts for the different methods and datasets automatically create an output folder with five runs inside the ```Forecasts``` folder. [Weights & Biases](https://wandb.ai/site) is used to log intermediate results.

To get the ETS, ARIMA, and THETA forecasts, run the ```*_baselines.R``` files.

To generate the results for the N-BEATS-S variants reported in the paper, run the different scripts for all the methods in the ```M3``` and ```M4``` folders. This will create 5  ```.csv``` files with forecasts for each method (note: newly generated forecasts can slightly deviate from the intermediate results due to random initializations). Alternatively, you can run the ```run_all_methods.py``` script, which runs all methods for one dataset. However, this will take a long time in terms of runtime (several days).

Next, put the folders with the output ```.csv``` files into the ```R scripts``` folder (see ```M*_evaluation.R``` for how these folders are read in). Running the ```M3_evaluation.R``` and ```M4_evaluation.R``` scripts will generate both the tables (Table 2 and Table A.1) and MCB plots (Figures 3, 4, A.1, and A.2). We provide our forecasts for the different methods in the ```R scripts``` folder via this [Google Drive](https://drive.google.com/file/d/1lBIW95MwPcHD5Pnl8mQP7t36G6LQWQGH/view?usp=drive_link).

To generate Figure 2, you need to first run ```M3_evaluation.R``` and ```M4_evaluation.R```. Then, ```pareto_plot.py``` will then automatically read the tables with results.

To generate Figure 5 download the ```lambda``` plot from a gradnorm run from ```wandb``` as a ```.csv``` and put it in the ```Graph_data``` folder. For Figure 6, do the same but for a GcosSim run, and a ```cosine``` plot. For Figure 7, 2 plots have to be downloaded from a ```wandb``` auxinash run: the ```lambda``` plot, and the ```hyperstep_p``` plot. After saving these, run the ```Discussion_graphs.py``` script to generate all the figures. If you only want to generate one, change the ```dataset_list``` and ```variable_list``` variables.

To generate Figure 8, run the ```Kappa_tuning.py``` script for the M3 and M4 datasets. Download the ```wandb``` ```M*_kappa_tuning``` project and save it in the ```Graph_data``` folder. To generate the figures, run the ```TARW_graphs.py``` script. We already provided our results in the ```Graph_data``` folder via this [Google Drive](https://drive.google.com/file/d/1lBIW95MwPcHD5Pnl8mQP7t36G6LQWQGH/view?usp=drive_link).

All models were trained on Xeon Gold 6140 CPU@2.3 GHz, 45 GiB RAM, and an NVIDIA P100 GPU. The runtimes of a single ```wandb``` run ranged from 45 minutes (for N-BEATS), and 2 hours 25 minutes (for Auxinash).
## Acknowledgements
Our code builds upon the code from [Van Belle et al. (2023)](https://github.com/VerbekeLab/n-beats-s).

To implement the dynamic loss weighting methods we used the following repos as a starting point:

- Lucas Bo Tang for GradNorm: ([https://github.com/LucasBoTang/GradNorm](https://github.com/LucasBoTang/GradNorm))
- Aviv Navon for Gradient Cosine Similarity ([https://avivnavon.github.io/AuxiLearn/](https://avivnavon.github.io/AuxiLearn/))
- The LibMTL library for uncertainty weighting ([https://github.com/median-research-group/LibMTL](https://github.com/median-research-group/LibMTL))
- The NashMTL/Auxinash repo from Aviv Shamsian ([https://github.com/AvivSham/auxinash])

Reference:

Van Belle, J., Crevits, R., & Verbeke, W. (2023). Improving forecast stability using deep learning. International Journal of Forecasting, 39(3), 1333-1350.

## Contact
Daan Caljon (daan.caljon@kuleuven.be)
