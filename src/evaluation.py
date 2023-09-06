import os
import sys
import numpy as np
import torch
import pandas as pd
import random
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import config
from dataset import Dataset
from data_augment import (
                get_validation_augmentation,
                get_preprocessing
                )
from utils import (
            visualize,
            visualise_predictions,
            plot_results)
from anova import anova_test
from adjustText import adjust_text
from models.deeplabv3.deeplabv3 import DeepLabV3Plus
# from models.u_netplusplus.u_netplusplus import UnetPlusPlus
from models.transfer_attention_u_effnet_plusplus.model_creator import SegmentationModel
from models.V2_Transfer_Attention_U_Effnet_PlusPlus.v2_u_effnet_tr_att import LightningModel


"""
This is the main file for further evaluation of different models by:
        1. Loading the different models from their last checkpoint
        2. Evaluating the model on the test sets
        3. Storing their predicted Dice Scores metrics into an array
        4. Applying ANOVA and Tukey HDS test on this array to identify if 
        there is a statistical significance between the models and also either 
        accept or reject the null hypothesis.
        5. Printing the results to the console by also providing the different p-values
        6. Further visualizations of the different models' predictions is performed using the 
        visualise_predictions function from the utils.py file.
"""

# Loading the differnet models form their last checkpoint 
# and storing them in a list named "obj".
deeplab = DeepLabV3Plus.load_from_checkpoint(config.deeplab["LAST_CKPT"])
# unetplusplus = UnetPlusPlus.load_from_checkpoint("/content/drive/MyDrive/src/Segmentation/models/u_netplusplus/weights/last.ckpt")
u_effnet_tr_att = SegmentationModel.load_from_checkpoint(config.transfer_attention_u_effnet_plusplus["LAST_CKPT"])
v2_u_effnet_tr_att = LightningModel.load_from_checkpoint(config.v2_transfer_attention_u_effnet_plusplus["LAST_CKPT"])
obj = {'DeepLabV3+':deeplab, 'Efficient Unet ATT TR':u_effnet_tr_att, 'Efficient Unet V2 ATT TR':v2_u_effnet_tr_att} #, 'Unet++':unetplusplus}


# Storing the different paths for:
# 1. The current score board directory for the DDR dataset
# 2. The test set directory with retinal images from the DDR dataset
# 3. The test set with segmentation masks from te DDR Dataset
# 4. The output folder to store the results
# If a different dataset is selected such as the FGADR dataset, 
# please then change the paths accordingly
ddr_csv_path = config.DDR_score_board_dir
ddr_x_test_dir = config.DDR_EX_dataset["x_test_dir"]
ddr_y_test_dir = config.DDR_EX_dataset["y_test_dir"]
output_results_path = config.Output_Result_Folder

ddr_df = pd.read_csv(ddr_csv_path)
final_DDR = ddr_df.sort_values(by=['test_mean_iou', 'test_mean_F1'], ascending=False, ignore_index=True)

# Initialize two dataset, the first is for inference 
# and the second is only to extract the unaugmented images
test_dataset = Dataset(
        ddr_x_test_dir, 
        ddr_y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(),
    )
test_dataset_vis = Dataset(
        ddr_x_test_dir, 
        ddr_y_test_dir, 
        augmentation=get_validation_augmentation(),
        )



# Plots a scatter plot for the models 
# and their specific average Dice and Intersection over Union Metrics
# Further arguments and documentation for this function are outlined in the utils.py file
plot_results(dataframe=final_DDR, 
             output_dir=output_results_path)

# Visualizes the predictions of the models for the test set, 
# the corresponding images and their ground truth masks are also visualized
for i in range(1):
  visualise_predictions(output_dir = output_results_path,
                        position=i+100,
                        dataset=test_dataset, 
                        dataset_vis = test_dataset_vis,
                        **obj)

# Applying the statistical evaluation function 
# on the test dataset and on all specified models in the obj list
# Further documentation about this function is providing in the anova.py file
infer_time_df_ddr = anova_test(dataset=test_dataset,
                                **obj)