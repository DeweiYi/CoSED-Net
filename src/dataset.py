import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):

    """
    This class is used to load the selected dataset and create the sub-sets  as 
    training, validation and testing.
    
    Args:
        images_folder (str): path to images folder
        masks_folder (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    Return:
        Image and associated mask within a given set 
        such as the training, validation or test set,
        under the form of a tuple
    
    """

    def __init__(
            self, 
            image_folder, 
            mask_folder,
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.augmentation = augmentation
        self.ids = os.listdir(image_folder)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __len__(self):
        # Returns the length of the database
        return len(self.ids)
    
    def __getitem__(self, key):
        # Allocates the image and mask paths
        image_files = os.path.join(self.image_folder, self.ids[key])
        mask_files = os.path.join(self.mask_folder, self.ids[key].replace(".jpg", "_segmentation.png"))

        # Reads the image
        image = cv2.imread(image_files)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reads the mask
        mask = cv2.imread(mask_files, 0)

        # Convers all pixels that correspond to 255 and are not 0 as 1
        # The rest remain 0 given that this is a binary mask
        mask = np.where((mask <= 255) & (mask != 0), 1, mask)
        mask = np.expand_dims(mask, axis=-1)
             
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask

