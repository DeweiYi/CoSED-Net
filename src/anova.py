import time
from scipy.stats import f_oneway
import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import random
from pprint import pprint
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from scipy import stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)
from tqdm.notebook import tqdm
from tqdm import tqdm
                
def anova_test(dataset, **models):

  """
  This function will run the ANOVA statical test on the specified dataset 
  and on all models in a specified dictionary object.
  Addionally, Tukey HSD test is applied to further evaluate the statical significance
  between the specified models by performing ttest for all combinations of the models in the dictionary object.
  The results would be displayed in the console in a table format.
  Furthermore, the inference time of the different models is also recorded for further analysis.

  Args:
    dataset: The dataset that the models will be evaluated on.
    models: The dictionary object that contains the models that will be evaluated.
  
  Returns:
    inference_time_df (DataFrame): The dataframe that contains the inference time of the different models.
  """
  
  # Creating empty arrays for both Dice Scores and inference time 
  # based on every image in the specified dataset
  # alongside two dataframes that would keep track of 
  # the inference time and the Dice Scores based on every inferred image
  f1_scores_model = np.empty(0)
  inference_time = np.empty(0)
  models_dict = pd.DataFrame()
  inference_time_df = pd.DataFrame()

  # Iterating over the models in the dictionary object
  for i, (name, model) in enumerate(models.items()):
    # Iterating over the images and segmentation masks in the dataset
      for i in tqdm(range(len(dataset))): #len(dataset)
        image, mask = dataset[i]
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)
        x_tensor = torch.from_numpy(image)
        
        # Running the inference on a given image
        with torch.no_grad():
            model.eval()

            # Monitoring the inference time 
            # with a start time and end time
            start_time = time.time()
            logits_mask = model(x_tensor)
            end_time = time.time()

            prob_mask = logits_mask.sigmoid()
            # Applying the threshold of 0.5 for the predicted segmentation mask
            pred_mask = (prob_mask > 0.5).float()

        # Calculating the different between the start time and end time 
        exact_time = round(end_time-start_time, 3) * 1000

        # Creating the TP, FP, FN and TN results for every image 
        # based on it corresponding ground truth and predicted segmentation mask
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        # Calculating the Dice Score based on the TP, FP, FN and TN results 
        image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # Appending the Dice Score to the f1_scores_model array
        f1_scores_model = np.round(np.append(f1_scores_model, image_f1), decimals=3)
        # Appending the inference time to the inference_time array
        inference_time = np.append(inference_time, exact_time)
      
      # Appending the inference time and the Dice Scores 
      # to the models_dict and inference_time_df dataframe
      models_dict[f"{name}"] = f1_scores_model
      inference_time_df[f"{name}"] = inference_time

      # Resetting the f1_scores_model and inference_time arrays
      # for the next model in the dictionary object and the next image in the dataset
      f1_scores_model = np.empty(0)
      inference_time = np.empty(0)

  # Applying the Tukey HSD test to the models_dict dataframe alongside the ANOVE test
  stacked_data = models_dict.stack().reset_index()
  stacked_data = stacked_data.rename(columns={'level_0': 'id','level_1': 'Model', 0:'F1_Score'})
  anova_test = anova_oneway(stacked_data['F1_Score'],
                            stacked_data['Model'], use_var="equal")
  MultiComp = MultiComparison(stacked_data['F1_Score'],
                            stacked_data['Model'])

  # Printing the Resutls from the statistical tests
  print(f"ANOVA P-value:{anova_test.pvalue}")
  print(MultiComp.tukeyhsd().summary())

  # Returning the inference_time_df dataframe
  return inference_time_df