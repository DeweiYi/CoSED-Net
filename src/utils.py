import matplotlib.pyplot as plt
import torch
from pathlib import Path
import random
import os
import sys
import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader
from src.dataset import Dataset
from src.data_augment import (
                get_training_augmentation,
                get_validation_augmentation,
                get_preprocessing
                )

def resutls_to_csv(csv_path, test_metrics, model_name):  
    """
    A method that takes the metric results 
    form the testing set and saves them to a csv file
    for a given model name 
    Args:
        csv_path (str): path to the csv file
        test_metrics (dict): dictionary containing the test metrics
        model_name (str): name of the model
    Return: 
        test_df (dict): dictionary containing the test metrics
    """
    # Creates a copy of the test_metrics dictionarory
    # to avoid changing the original one
    scores_dict = test_metrics[0].copy()

    # Goes though the dictionary and rounds 
    # and transformes the metrics into percentages
    for key in scores_dict:
        scores_dict[key] = round(scores_dict[key], 4)
        scores_dict[key] = scores_dict[key]*100

    # Add the name of the models to the dictionary
    scores_dict['model_name'] = model_name

    # Creating a variable to store the csv file path
    score_board_dir = csv_path

    # Creating a temporary directory to reindex 
    # the "scores_dict" variable 
    temp_df = pd.DataFrame(scores_dict, index=[0])
    columns_titles = ["model_name", "test_loss", "test_mean_accuracy",	"test_mean_recall", "test_mean_precision", "test_mean_iou", "test_mean_F1"]

    # Creating a new Dataframe with the new columns titles
    # and saving the results to the csv file
    df = pd.read_csv(score_board_dir).reindex(columns=columns_titles)
    df = pd.concat( [df, temp_df], axis=0, ignore_index=False)
    df.to_csv(score_board_dir, index=False)

    test_df = pd.read_csv(score_board_dir)
    print("Results Saved to {}".format(score_board_dir))
    return test_df

def visualize(**images):
    """
    A method that visualizes the predicted masks of the different models
    
    Args:
        filename (str): name of the file to save the visualized images
        images (dict): dictionary containing the images to visualize
    """
    
    # The number of images to visualize
    n = len(images)
    # Selecting the size of the figure
    plt.figure(figsize=(10, 5))
    # Creating a grid of the images in a subplot format
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), fontsize=20)
        plt.imshow(image)
    # Adusting the text and titles of the different predicted masks
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.03, 
                    hspace=0.4)
    # Saving the figure with the specified file name
    # given by the "filename" variable
    # plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

def visualise_predictions(output_dir, position, dataset, dataset_vis, **models):
    """
    A method that predicts/inferse the sementation mask for a given retinal image
    based on all specified models. Further visualization is addapeted using the 
    "visualize(filename, **images)" function in the "utils.py" file
    
    Args:
        output_dir (str): path to the output directory
        position (int): position of the image in the dataset
        dataset (Dataset): the specified test set dataset for making the predictions
        with specified data augmentation and preprocessing
        dataset_vis (Dataset): the specified test set dataset 
        to visualize the images that are being infreded 
        models (dict): dictionary containing the different neural network models
    """

    # Creating a dictionary to store the predicted masks
    model_dict = {}

    # Going through the different models for inference
    # by also specifying the retinal image
    for i in range(1):
        image_vis = dataset_vis[position][0].astype('uint8')
        image, mask = dataset[position]
        
        # Going through the different models
        for i, (name, model) in enumerate(models.items()):
            # Creating a tensor that holds the image array
            x_tensor = torch.from_numpy(image)
            # Predicting the lesion segmentation mask
            with torch.no_grad():
                model.eval()
                logits_mask = model(x_tensor)
                pred_masks = logits_mask.sigmoid().squeeze()
            # Applying the threshold of 0.5 to distinguish between
            # the lesion and the background
            output_mask = (pred_masks>0.5).int()
            mask = mask.squeeze()
            # Adding the predicted mask to the dictionary
            model_dict[f"{name}"] = output_mask

        # Visualization of the predicted masks
        # alongside the ground truth mask and the retinal image
        visualize(
            filename=f"{output_dir}\DDR_Pred_Vis",
            image=image_vis, 
            ground_truth_mask=mask,
            **model_dict
        )
        # The model dictionary is emptied for further predictions 
        model_dict = {}

def plot_results(dataframe, output_dir):

    """
    A method that plots the results from all models made on the test set of the DDR dataset
    in a scatter plot format. The results are plotted for the the Average Dice Score Metric 
    and the Average Intersection Over Union metric.
    
    Args:
        dataframe (Dataframe): dataframe containing the results from the testing set for all Models
        output_dir (str): path to the output directory to save the figure
    """

    # Assigning an empty array that would gold all the different models' names and metrics
    ts = []
    # Storing only the models' names and their average Dice score and IOU metrics
    model_names = dataframe["model_name"].tolist()
    mean_iou = dataframe["test_mean_iou"].tolist()
    mean_f1 = dataframe["test_mean_F1"].tolist()

    # Assigning random colors for every model
    colors = np.random.rand(len(model_names))
    # Creating a figure with the specified size
    plt.figure(figsize=(18,9))
    # Creating a scatter plot with the specified parameters
    plt.scatter(mean_f1, mean_iou, s=200, c=colors)
    plt.xlabel("Average Dice Score", fontsize=11)
    plt.ylabel("Average Intersection Over Union", fontsize=11)
    plt.title("Scatter Plot of the Different Models \n Based on their Dice Scores and IoU")

    # Going through the different models and adding their names and metrics to the array
    for i, label in enumerate(model_names):
        ts.append(plt.text(mean_f1[i], mean_iou[i], label, fontsize=12))

    # Adding additional arrows for further clarity 
    adjust_text(ts, x=mean_f1, y=mean_iou, force_points=4, arrowprops=dict(arrowstyle='->', 
    color='black'), autoalign=True)
    plt.grid()

    # Saving the figure with the specified output directory
    plt.savefig(f"{output_dir}\seg_scaterplot.png", bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()

def gpu_checker():
    """
    A method that checks if the GPU is available
    """
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("CPU will be used. This is not advised")

def set_seed(seed = 42):
    '''
    Seting the seed of the entire training process to 42
    so  that the outcome of the results are the same every time it is run. 
    This is done for reproducibility purposes.
    The code has been addapeted from 
    https://www.kaggle.com/code/debarshichanda/efficientnetv2-mixup-leak-free/notebook

    Args:
        seed (int): seed for the random number generator
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("The random seed has been reset")
    
def get_last_checkpoint(last_ckpt_dir=None):
    """
    A method that returns the last checkpoint of the training process 
    for a specified model.

    Args:
        last_ckpt_dir (str): path to the directory where the last checkpoint is stored
    """
    if last_ckpt_dir != None and Path(last_ckpt_dir).exists():
        return last_ckpt_dir
    return None


def get_checkpoint_callback(ckpt_dir, 
                            ckpt_name,
                            monitor='val_mean_iou',
                            mode='max',
                            save_last=True):
    """
    A method that returns a callback that saves the models weights and hyperparameters.
    This checkpoint callback is triggered when the monitored quantity is improved.
    In this case we are monitoring if there is substantial higher improvement in the IoU metric.

    Args:
        ckpt_dir (str): path to the directory where the checkpoint will be saved
        ckpt_name (str): name of the checkpoint
        monitor (str): name of the metric that is being monitored
        mode (str): mode of the monitored metric, it could be either 'max' or 'min'
        save_last (bool): whether to save the last checkpoint or not

    Returns:
        checkpoint_callback (Callback): ModelCheckpoint callback object
    """

    return ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor,
        filename=ckpt_name,
        mode=mode,
        save_last=save_last
    )

def get_early_stop_callback(monitor="val_mean_iou", 
                            patience=5, 
                            mode="max"):

    """
    A method that returns a callback that stops the training process of a specified models
    when the monitored quantity is not improved for a specified number of epochs.
    In this case, the monitored quantity is the IoU metric and the patience is set to 5 epochs.
    After 5 epochs without improvement, the training process is stopped.

    Args:
        monitor (str): name of the metric that is being monitored
        patience (int): number of epochs without improvement before the training process is stopped
        mode (str): mode of the monitored metric, it could be either 'max' or 'min'

    Returns:
        early_stop_callback (Callback): EarlyStopping callback object
        
    """
    return EarlyStopping(monitor=monitor, 
                        patience=patience, 
                        verbose=False, 
                        mode=mode)

def get_data_loaders(x_train_dir, 
                     y_train_dir, 
                     x_valid_dir, 
                     y_valid_dir, 
                     x_test_dir, 
                     y_test_dir,
                     train_batch=4,
                     valid_batch=2,
                     test_batch=1,
                     num_workers=2):
    """
    A method that returns the data loaders for the training, validation and testing sets for a given dataset.
    The data loaders are created using the specified batch sizes and number of workers.
    All of the datasets are formed by loading the images and masks from the specified directories.
    Additionally, specific transformations are applied to the images and masks such as 
    data augmentation and pre-processing.

    Args:
        x_train_dir (str): path to the directory where the training images are stored
        y_train_dir (str): path to the directory where the training masks are stored
        x_valid_dir (str): path to the directory where the validation images are stored
        y_valid_dir (str): path to the directory where the validation masks are stored
        x_test_dir (str): path to the directory where the testing images are stored
        y_test_dir (str): path to the directory where the testing masks are stored
        train_batch (int): batch size for the training set
        valid_batch (int): batch size for the validation set
        test_batch (int): batch size for the testing set
        num_workers (int): number of workers for the data loaders

    Returns:
        train_loader (DataLoader): data loader for the training set
        valid_loader (DataLoader): data loader for the validation set
        test_loader (DataLoader): data loader for the testing set
    """
    # Loading the images and masks from the training set
    train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(),
    )

    # Loading the images and masks from the validation set
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(),
    )

    # Loading the images and masks from the testing set
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(),
    )

    # Creating the data loaders for the training, validation and testing sets
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def get_fgadr_loaders(x_train_dir, 
                     y_train_dir,
                     x_test_dir, 
                     y_test_dir,
                     train_batch=4,
                     test_batch=1,
                     num_workers=2):
                     
    # Loading the images and masks from the training set
    train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(),
    )

    # # Loading the images and masks from the validation set
    # valid_dataset = Dataset(
    #     x_valid_dir, 
    #     y_valid_dir, 
    #     augmentation=get_validation_augmentation(), 
    #     preprocessing=get_preprocessing(),
    # )

    # Loading the images and masks from the testing set
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(),
    )

    # Creating the data loaders for the training, validation and testing sets
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
    # valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader