import numpy as np
import torch
from pprint import pprint
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp

import config
from models.model_creator import SegmentationModel
import json
import gc
# gc.collect()
# torch.cuda.empty_cache()

from utils import (
            visualize,
            gpu_checker,
            set_seed,
            resutls_to_csv,
            get_last_checkpoint,
            get_checkpoint_callback,
            get_early_stop_callback,
            get_data_loaders)

"""
This is the main file that will run the training and validation of the model.
It will also save the model and the training history to the specified folder
for further possibility to apply TensorBoard.
Addionally, the model would be evaluated on the test set from a specified dataset 
and its results would be saved to the specified csv file for further analysis.

"""


# Initial Checks using the set_seed function to reset the random seed 
# and the gpu_checker function to check if there is a GPU available.
set_seed()
gpu_checker()

# Assign the specific score board path from the config.py file for the DDR dataset.
# if chosen a different dataset change this directory accordingly.
resutls_csv_dir = config.DDR_score_board_dir



"""
  Model Configuration
"""

# Assign the spesifed model's configuration from the config.py file. 
# to the model_setup variable 
# If you want to use a different model change this accordingly.
model_setup = config.transfer_attention_u_effnet_plusplus

# Specifying  the model's name
NAME = model_setup["NAME"]

# Specifying the model's basline architecture
# For example, if you want to use the U-Net architecture,
# you would should specify that under the model's configuration in the config.py file
ARCH = model_setup["ARCH"]

# Specifying the model's specific Segmentation Models Pytorch name
SMP_MODEL_NAME = model_setup["SMP_MODEL_NAME"]

# Specifying if the model should use pretrained weights or not
PRETRAINED = model_setup["PRETRAINED"]

# Specifying if the model should use 
# the Special and Channel "Squeeze and Exitstation" (scSE) block or not
ATTENTION = model_setup["ATTENTION"]

# Specifying the directory to log the traing process
LOGGER_DIR = model_setup['LOGGER_DIR']
LOGGED_DATA = f"{LOGGER_DIR}/{NAME}_logs"

# Specifying the directory to save the model's weights and hyperparameters
CKPT_DIR = model_setup["CKPT_DIR"]
CKPT_NAME = model_setup["CKPT_NAME"]
LAST_CKPT_DIR = model_setup["LAST_CKPT_DIR"]


# Additional configurations

# FAST_DEV_RUN is used to specify 
# if the Trainer should only run for one training and validation epoch
FAST_DEV_RUN=True

# Specifying the number of epochs for the training
NUM_EPOCHS=50

# Specifying the size of the training mini-batch
TRAIN_BATCH=2
# Specifying the size of the validation mini-batch
VAL_BATCH=1

# Specifying the size of the testing mini-batch
TEST_BATCH=1

# Specifying the number of workers for the data loader
# here the number of workers is set to 0 given the lower 
# computation resources of the local machine
NUM_WORKERS=0


# Specifying the chosen dataset
# if another dataset is selected, then change this variable accordingly
# based on the specific dataset's configuration in the config.py file
dataset = config.DDR_EX_dataset


"""
  Model Initialisation
"""

# Initialisation of the checkpoint callback functions
checkpoint = get_checkpoint_callback(ckpt_dir=CKPT_DIR, 
                                     ckpt_name=CKPT_NAME)
last_checkpoint = get_last_checkpoint(LAST_CKPT_DIR)

# Initialisation of the diffrent dataloaders from the get_data_loaders function in the utils.py file
# based on the dataset's configuration in the config.py file 
# followed by the different batch sizes and number of workers
train_loader, valid_loader, test_loader = get_data_loaders(dataset['x_train_dir'], 
                                                           dataset['y_train_dir'],
                                                           dataset['x_valid_dir'],
                                                           dataset['y_valid_dir'], 
                                                           dataset['x_test_dir'], 
                                                           dataset['y_test_dir'],
                                                           train_batch=TRAIN_BATCH,
                                                           valid_batch=VAL_BATCH,
                                                           test_batch=TEST_BATCH,
                                                           num_workers=NUM_WORKERS)


# Creating the neural network model using the SegmentationModel class
# imported from models.transfer_attention_u_effnet_plusplus.model_creator file
# and the model's configuration from the config.py file such as the following
# architecture, pretrained weights, attention block, encoder's name, etc.
# For additional class arguments please refer to the model_creator in the models directory folder
model = SegmentationModel(arch=ARCH, 
                    encoder_name=SMP_MODEL_NAME,
                    pretrained=PRETRAINED,
                    attention_type=ATTENTION)

# Create the additional Tensor Board Logger
logger = TensorBoardLogger(save_dir=LOGGED_DATA, 
                           name=NAME, 
                           )


"""
  Model Training
  The Trainer class is initialized with the following arguments:

  Args: 
    gpus: number of GPUs to use
    max_epochs: number of epochs to train and validate
    fast_dev_run: if True, only train for one epoch
    resume_from_checkpoint: if True, resume training from last checkpoint
    callbacks: list of callbacks to use
    checkpoint_callback: checkpoint callback to use
    logger: logger to use (TensorBoardLogger)
"""

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=NUM_EPOCHS,
    fast_dev_run=FAST_DEV_RUN,
    resume_from_checkpoint=last_checkpoint,
    callbacks=[checkpoint],
    logger=logger
)

# The Trainer class is fitter to the specified training 
# and validation dataset using the specified dataloaders 
# and the specified model
trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=valid_loader,
)


"""
  Model Testing

  The following code will test the model on the test dataset
  and save the results to the specified csv file.

  The test function is used from the Trainer class with the following arguments:

  Args: 
    model: model to test
    dataloaders: dataloaders to test on
    verbose: if True, print the results
"""
test_metrics= trainer.test(model, 
                           dataloaders=test_loader,
                           verbose=True)

# The average metric results are saved to the specified csv file
resutls_to_csv(resutls_csv_dir, test_metrics, NAME)