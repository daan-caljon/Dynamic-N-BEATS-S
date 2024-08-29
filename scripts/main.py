import gc
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import random
import os
import sys
import copy
import matplotlib.pyplot as plt

from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
#Set directory
DIR = r""
os.chdir(DIR)
sys.path.append(DIR)

import src.data.Read_data as read_data
from src.methods.Learner import StableNBeatsLearner

dataset = "M3_Monthly"
trainset, valset, testset = read_data.read_data(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

#setup
wandb_project_name = 'nbeats_stability_m4_monthly'
job_type_name = 'test' 

# one of:
# - 'test', 
# - 'validation_full' --> e.g., for lambda value tuning
# - 'validation_earlystop' --> for specifying number of epochs


hyperparameter_defaults = dict()
if dataset == "M3_Monthly":
    hyperparameter_defaults["backcast_length_multiplier"] = 6
    hyperparameter_defaults["LH"] = 20 
elif dataset == "M4_Monthly":
    hyperparameter_defaults["backcast_length_multiplier"] = 4
    hyperparameter_defaults["LH"] = 10

#Change depending on setting (see paper)
hyperparameter_defaults['epochs'] = 4000  
hyperparameter_defaults['learning_rate'] = 0.00001 

#Dynamic weighting
hyperparameter_defaults["balance_type"] = "rw" #["gradnorm", "no", "weighted gcossim", "gcossim","rw"]
#If you use rw, also change lambda_cap (kappa in paper)
hyperparameter_defaults["alpha"] = 1 #hyperparameter for gradnorm
hyperparameter_defaults["learning_rate_gradnorm"] = 0.0025 #hyperparameter for gradnorm 
hyperparameter_defaults["lambda_cap"] = 0.35 #lambda_cap = 1 is random weighting
hyperparameter_defaults["lambda"] = 0.15 #only when using static weighting


#Same for all settings
hyperparameter_defaults['batch_size'] = 512 #512 
hyperparameter_defaults['nb_blocks_per_stack'] = 1
hyperparameter_defaults['thetas_dims'] = 256 #256 
hyperparameter_defaults['n_stacks'] = 20  #20 
hyperparameter_defaults['share_weights_in_stack'] = False
hyperparameter_defaults['hidden_layer_units'] = 256 #256
hyperparameter_defaults['share_thetas'] = False
hyperparameter_defaults["dropout"] = False
hyperparameter_defaults["dropout_p"] = 0.0
hyperparameter_defaults["neg_slope"] = 0.00
hyperparameter_defaults["weight_decay"] = 0.00
hyperparameter_defaults["rndseed"] = 2000
hyperparameter_defaults["loss_function"] = 1 # 1 == RMSSE / 2 == RMSSE_m / 3 == SMAPE / 4 == MAPE
hyperparameter_defaults["shifts"] = 1
hyperparameter_defaults['patience'] = 2000 # Only affects 'validation_earlystop' runs



if job_type_name == 'test':
    is_val = False
    do_earlystop = False
    m4_train, m4_eval = valset, testset
elif job_type_name == 'validation_full':
    is_val = True
    do_earlystop = False
    m4_train, m4_eval = trainset, valset
elif job_type_name == 'validation_earlystop':
    is_val = True
    do_earlystop = True
    m4_train, m4_eval = trainset, valset

def sweep_function():
    wandb.init(config = hyperparameter_defaults,
               project = wandb_project_name,
               job_type = job_type_name)
    config = wandb.config
    run_name = wandb.run.name
    
    # Initialize model
    StableNBeats_model = StableNBeatsLearner(device, 6, config) #length of forecast

    # Train & evaluate
    forecasts_df_m4m = StableNBeats_model.train_net(m4_train, m4_eval, 13, is_val, do_earlystop) #13 is forigins
    # Save forecasts
    forecasts_df_m4m.to_csv('m4m_nbeats_stability_' + job_type_name + '_' + run_name + '.csv', index = False)
    #forecasts_df_m4m.to_csv('/content/drive/My Drive/Colab Notebooks/Sweeps/m4m_nbeats_stability_' + job_type_name + '_' + run_name + '.csv', index = False)

sweep_config = {
    "name": "sweep",
    "method": "grid",
    "parameters": {
        "rndseed": {
            "values": [2000] #2000,4000,6000,8000,10000

    }
}}
sweep_id = wandb.sweep(sweep_config, project = wandb_project_name)
wandb.agent(sweep_id, function = sweep_function)