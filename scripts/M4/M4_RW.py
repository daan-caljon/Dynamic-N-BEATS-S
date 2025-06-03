import os
import sys

import torch

import wandb

# Set directory
DIR = os.path.dirname(os.path.abspath(__file__))
# go two directories up
DIR = os.path.dirname(os.path.dirname(DIR))
os.chdir(DIR)
sys.path.append(DIR)

import src.data.Read_data as read_data
from src.methods.Learner import StableNBeatsLearner

wandb.login()

dataset = "M4_Monthly" # "M4_Monthly" or "M3_Monthly"
trainset, valset, testset = read_data.read_data(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

# setup
wandb_project_name = "Change_name"
job_type_name = "test"

# one of:
# - 'test',
# - 'validation_full' --> e.g., for lambda value tuning


hyperparameter_defaults = dict()
if dataset == "M3_Monthly":
    hyperparameter_defaults["backcast_length_multiplier"] = 6
    hyperparameter_defaults["LH"] = 20
elif dataset == "M4_Monthly":
    hyperparameter_defaults["backcast_length_multiplier"] = 4  #

    hyperparameter_defaults["LH"] = 10  # 10

#Depending on the wandb config (see below), these hyperparameters will be overwritten
# Change depending on setting (see paper)
hyperparameter_defaults["epochs"] = 4000
hyperparameter_defaults["learning_rate"] = 0.00001

# Dynamic weighting
hyperparameter_defaults["balance_type"] = (
    "auxinash"  # ["gradnorm", "no", "weighted gcossim", "gcossim","rw","nashmtl","auxinash"]
)
# If you use rw, also change lambda_cap (kappa in paper)
hyperparameter_defaults["alpha"] = 1  # hyperparameter for gradnorm
hyperparameter_defaults["learning_rate_gradnorm"] = (
    0.0025  # hyperparameter for gradnorm
)
hyperparameter_defaults["lambda_cap"] = 1  # lambda_cap = 1 is random weighting
hyperparameter_defaults["lambda"] = 0.15  # only when using static weighting
# nashmtl
hyperparameter_defaults["optim_niter"] = 20
hyperparameter_defaults["update_weights_every"] = 1
hyperparameter_defaults["max_norm"] = 1
hyperparameter_defaults["scale_alpha"] = True
# auxinash
hyperparameter_defaults["preference_accuracy"] = (
    0.9  # This is the preference for accuracy (from 0 to 1)
)
hyperparameter_defaults["auto_p"] = True
hyperparameter_defaults["hyperstep"] = 10
hyperparameter_defaults["preference_lr"] = 0.0001

# Same for all settings
hyperparameter_defaults["batch_size"] = 512  # 512
hyperparameter_defaults["nb_blocks_per_stack"] = 1
hyperparameter_defaults["thetas_dims"] = 256  # 256
hyperparameter_defaults["n_stacks"] = 20  # 20
hyperparameter_defaults["share_weights_in_stack"] = False
hyperparameter_defaults["hidden_layer_units"] = 256  # 256
hyperparameter_defaults["share_thetas"] = False
hyperparameter_defaults["dropout"] = False
hyperparameter_defaults["dropout_p"] = 0.0
hyperparameter_defaults["neg_slope"] = 0.00
hyperparameter_defaults["weight_decay"] = 0.00
hyperparameter_defaults["rndseed"] = 2000
hyperparameter_defaults["loss_function"] = (
    1  # 1 == RMSSE / 2 == RMSSE_m / 3 == SMAPE / 4 == MAPE
)
hyperparameter_defaults["shifts"] = 1
hyperparameter_defaults["patience"] = 2000  # Only affects 'validation_earlystop' runs


if job_type_name == "test":
    is_val = False
    do_earlystop = False
    m4_train, m4_eval = valset, testset
elif job_type_name == "validation_full":
    is_val = True
    do_earlystop = False
    m4_train, m4_eval = trainset, valset
elif job_type_name == "validation_earlystop":
    is_val = True
    do_earlystop = True
    m4_train, m4_eval = trainset, valset

# wandb.init(config = hyperparameter_defaults,
#               project = wandb_project_name,
#                 job_type = job_type_name)
# StableNBeats_model = StableNBeatsLearner(device, 6, hyperparameter_defaults) #length of forecast

# # Train & evaluate
# forecasts_df_m4m = StableNBeats_model.train_net(m4_train, m4_eval, 13, is_val, do_earlystop) #13 is forigins


def sweep_function():
    wandb.init(
        config=hyperparameter_defaults,
        project=wandb_project_name,
        job_type=job_type_name,
    )
    config = wandb.config
    run_name = wandb.run.name

    # Initialize model
    StableNBeats_model = StableNBeatsLearner(device, 6, config)  # length of forecast

    # Train & evaluate
    forecasts_df_m4m = StableNBeats_model.train_net(
        m4_train, m4_eval, 13, is_val, do_earlystop
    )  # 13 is forigins
    # Save forecasts
    if config["balance_type"] == "no" and config["lambda"] == 0:
        balance_type = "NBEATS"
    elif config["balance_type"] == "no" and config["lambda"] == 0.025:
        balance_type = "NBEATSS_low"
    elif config["balance_type"] == "no":
        balance_type = "NBEATSS_high"
    elif config["balance_type"] == "rw" and config["lambda_cap"] == 0:
        balance_type = "rw"
    elif config["balance_type"] == "rw" and config["lambda_cap"] == 0.125:
        balance_type = "TARW_low"
    elif config["balance_type"] == "rw" and config["lambda_cap"] == 0.20:
        balance_type = "TARW_high"
    else: 
        balance_type = config["balance_type"]
    file_name = dataset + "_" + job_type_name + "_" + balance_type + "_"+ run_name + ".csv"
    file_folder_name = os.path.join("Forecasts", dataset + "_" + balance_type + "_" + job_type_name)
    #check if folder exists
    if not os.path.exists(file_folder_name):
        os.makedirs(file_folder_name)
    #save file in folder
    file_name = os.path.join(file_folder_name, file_name)

    forecasts_df_m4m.to_csv(
        file_name, index=False
    )
    # forecasts_df_m4m.to_csv('/content/drive/My Drive/Colab Notebooks/Sweeps/m4m_nbeats_stability_' + job_type_name + '_' + run_name + '.csv', index = False)


sweep_config = {
    "name": "sweep",
    "method": "grid",
    "parameters": {
        "rndseed": {
            "values": [2000,4000,6000,8000,10000]  # 2000,4000,6000,8000,10000
        },
        "balance_type": {"values": ["rw"]},
        "lambda_cap": {"values": [1]},
        "learning_rate": {"values": [0.0001]},  # 0.0005,0.001,0.0025,0.005,0.01
        "epochs": {"values": [200]},  # 4000,6000,8000,10000,12000
    },
}
sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
wandb.agent(sweep_id, function=sweep_function)