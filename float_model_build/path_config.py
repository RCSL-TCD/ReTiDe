import torch
import os
from datetime import datetime


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Paths (PNG Version)
dataset_D10 = "/data/clement/data/DAVIS_training_10"
dataset_B10 = "/data/clement/data/bandon_training_10"
dataset_D20 = "/data/clement/data/Davis_training_20"
dataset_B20 = "/data/clement/data/bandon_training_20"
dataset_D30 = "/data/clement/data/davis_training_30"
dataset_B30 = "/data/clement/data/bandon_training_30"
dataset_D40 = "/data/clement/data/davis_training_40"
dataset_B40 = "/data/clement/data/bandon_training_40"
dataset_D50 = "/data/clement/data/DAVIS_training_50"
dataset_B50 = "/data/clement/data/bandon_training_50"

# Commented-out: Numpy dataset paths
# dataset_D10 = "/data/clement/data/DAVIS_training_10_np"
# dataset_B10 = "/data/clement/data/bandon_training_10_np"
# dataset_D20 = "/data/clement/data/Davis_training_20_np"
# dataset_B20 = "/data/clement/data/bandon_training_20_np"
# dataset_D30 = "/data/clement/data/davis_training_30_np"
# dataset_B30 = "/data/clement/data/bandon_training_30_np"
# dataset_D40 = "/data/clement/data/davis_training_40_np"
# dataset_B40 = "/data/clement/data/bandon_training_40_np"
# dataset_D50 = "/data/clement/data/DAVIS_training_50_np"
# dataset_B50 = "/data/clement/data/bandon_training_50_np"

# Testing Dataset Paths
test_dataset_patha = "/data/clement/data/market_gaussian_10_np"
test_dataset_pathb = "/data/clement/data/market_gaussian_20_np"
test_dataset_pathc = "/data/clement/data/market_gaussian_30_np"
test_dataset_pathd = "/data/clement/data/market_gaussian_40_np"
test_dataset_pathe = "/data/clement/data/market_gaussian_50_np"

benchmark_dataset_patha = "/data/clement/data/test_set_10_np/*/"
benchmark_dataset_pathb = "/data/clement/data/test_set_20_np/*/"
benchmark_dataset_pathc = "/data/clement/data/test_set_30_np/*/"
benchmark_dataset_pathd = "/data/clement/data/test_set_40_np/*/"
benchmark_dataset_pathe = "/data/clement/data/test_set_50_2_np/*/"

local_30_training = "/home/bledc/Videos/training_30"
local_30_training2 = "/home/bledc/Videos/bandon2_training_30"
# Model Checkpoint Paths


# Commented-out: Alternative Model Paths
wiener_path = "/data2/clement/models/16x16_halfprecision_thirdoverlap_refined_WINDOW00_13_44/model_epoch_1380.pt"  # nonclamp previous best benchmark
# wiener_path  = "/data/clement/models/stage1_noisepred_4_2001_52_59/model_epoch_380.pt"  # fine-tuned model
wiener_path = "/data/clement/models/best_nonblind_sig40_16_third/model_epoch_1440.pt"
# noise pred refine experiments
# wiener_path = "/home/bledc/data_dir/clement/models/wiener_frozenstd_blin/model_epoch_4.pt" #not actually 4 # noise predictor
# wiener_path = "/data/clement/models/tiny_pvv_net_dataloader_fixed21_19_25/model_epoch_80.pt" # pvv tiny/ medium pred noise
# wiener_path = "/data/clement/models/single_dense_neuron_k_net_resume14_25_37/model_epoch_200.pt" single neuron with noise pred I think 1x1conv elu
# wiener_path = "/data/clement/models/single_dense_neuron_k_net00_06_25/model_epoch_1000.pt" # SINGLE NEURON PB
# wiener_path = "/data/clement/models/single_dense_neuron_k_net_resume_test_png23_05_14/model_epoch_2220.pt" # name incorrect
# wiener_path = "/data/clement/models/single_linear_again_elu_datasetcheck_01_42_17/model_epoch_2220.pt" # name incorrect # predictive std net also.

# wiener_path = "/data/clement/models/single_linear_again_elu_datasetcheck_RealGTSTD23_50_37/model_epoch_1920.pt" # real frozen single node no noise pred.
benchmark_save_path = "/data2/clement/benchmarks"


# Pretrained Windows Paths
win1_path = "./unclamped/win1_unclamped.pt"
win2_path = "./unclamped/win2_unclamped.pt"

# Experiment name
experiment_name = "hw_denoiser_grayscale_3stage_addition_resblocks"

test_name =  "hw_denoiser_grayscale_3stage_addition_resblocks"
# Output Directory
def get_experiment_dir():
    """Generate an experiment directory with timestamp."""
    current_time = datetime.now().strftime("%H_%M_%S")
    path = f"/data/clement/models/{experiment_name}{current_time}"
    return path, current_time
