import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad, _calc_metrics, copy_Files
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from models.model import base_Model
import random

# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=132, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='sleepEDF', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default='home_dir', type=str,
                    help='Project home directory')
parser.add_argument('--config_version', default="sleepEDF", type=str,
                    help='Config choices: sleepEDF, sleepEDF_mini, sleepEDF_x, sleepEDF_modified')
parser.add_argument('--fine_tune_strategy', default="normal", type=str,
                    help='Choices: normal, partition')


args = parser.parse_args()


device = torch.device(args.device) # CPU or Cuda
experiment_description = args.experiment_description # Name for experiment
data_type = args.selected_dataset # Which dataset -> sleepEDF by default
method = 'TS-TCC'
training_mode = args.training_mode # Which mode of training, see options above
run_description = args.run_description # Name for run
config_version = args.config_version # Which configuration to use
fine_tune_strategy = args.fine_tune_strategy # Normal fine tuning or partition fine tuning

logs_save_dir = args.logs_save_dir # Directory where to save everything
os.makedirs(logs_save_dir, exist_ok=True) # Makes the directory, exist_ok=True allows for existing directory to be overwritten

exec(f'from config_files.{config_version}_Configs import Config as Configs')

configs = Configs()
configs.run_description = run_description # Used to modify the augmentations

###### fix random seeds for reproducibility ########
SEED = args.seed # Defaults to 132
torch.manual_seed(SEED) # Sets the seed for pytorch operations
# CuDNN is a GPU-accelerated library from NVIDIA
torch.backends.cudnn.deterministic = False # Operations might not be deterministic as it is False, setting to True might impact performance 
# If True, will perform analysis of system to optimize which algortihms to use, is only deterministic when input size are constant, setting to False avoids variability acorss runs due to different selection of algorithms
torch.backends.cudnn.benchmark = False 
np.random.seed(SEED) # Sets the seed for numpy operations 
random.seed(SEED)
#####################################################

# os.path.join(.) will take strings as inputs and concatenate them into one appropriate path, using separators of the operating system being used (i.e. is portable across devices)
if fine_tune_strategy == "partition":
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"{data_type}")
else:
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")

# --> will return a path such as: "C:\Users\denna\ThesisCode\logs_save_dir\experiment_description\run_description\training_mode + f"_seed_{SEED}""
os.makedirs(experiment_log_dir, exist_ok=True) # Creates the dictionary with path name above, makes all intermediate dictionaries. exist_ok=True allows for existing dictionaries to be overwritten

# Logging
# To keep track of the experiment and be able to uncover any bugs/mistakes in the code
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
# data_path = f".\data\{data_type}"
data_path = os.path.join(".", "data", data_type)

train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, logger) # See function in dataloader.py
# --> returns 3 objects, each object is an iterable pytorch DataLoader object, one iteration gives a batch of 128 30-second epochs 
logger.debug("Data loaded ...")

# Load models
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)


### Prepare the model if training mode is fine_tune, train_linear, or random_init ###
if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # Remove the linear classifer layer at the end of the encoder
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    # >> Returns a ordered dictionary

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # >> Returns a dictionary with weight matrices as values, contains convolution blocks and linear layer in the end
    # This dictionary is the same as model_dict

    # Deletes the linear layer in the end
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)

    # Instantiate the model with its weight matrices
    model.load_state_dict(model_dict)
    # Freeze everything except last layer
    set_requires_grad(model, pretrained_dict, requires_grad=False) # See utils set_requires_grad(.)

if training_mode == "random_init":
    model_dict = model.state_dict()

    # Delete all the parameters except for logits <<<<<<<<<<<<< This is not what appears to be happening here
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.



model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Creates a copy of the .py files used for this model (only for the self-supervised scenario)
if training_mode == "self_supervised":
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type, config_version)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)


# ------ Changed dataset to validation to avoid overfitting to test set during experimentation ------- #
if training_mode != "self_supervised":
    # Test model, generate classification report and confusion matrix
    outs = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
    _, _, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")