"""
Results Path
This is the paths where the results will be saved based on the diffrent dataset that have been selected
"""
DDR_score_board_dir = r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\results\DDR_score_board.csv"
FGADR_score_board_dir = "/content/drive/MyDrive/src/Segmentation/results/FGADR_score_board.csv"
Output_Result_Folder = r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\results"

"""
DDR Dataset Paths
This dictionary contains the paths to the DDR datasets that are used 
to construct the Dataset objects class that would be further used to create the different dataloaders

The following DDR_EX_dataset directory structure is as follows:
  x_train_dir = path to the training set with the retinal images
  y_train_dir = path to the training set with the segmentation masks
  x_val_dir = path to the validation set with the retinal images
  y_val_dir = path to the validation set with the segmentation masks
  x_test_dir = path to the test set with the retinal images
  y_test_dir = path to the test set with the segmentation masks
"""
DDR_EX_dataset = {
"x_train_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\train\image",
"y_train_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\train\label\EX",

"x_valid_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\valid\image",
"y_valid_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\valid\label\EX",

"x_test_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\test\image",
"y_test_dir": r"D:\Datasets\DDR-dataset\DDR-dataset\lesion_segmentation\test\label\EX",
}


"""
FGADR Dataset Paths

The following 4 directories: FGADR_EX_dataset, FGADR_MA_dataset, FGADR_HE_dataset, FGADR_SE_dataset
all speficy the paths to the FGADR datasets but the different is that they correspond to the different
lesion types.

EX - Hard Exudates
MA - Microaneurysms
HE - Hemorrhages
SE  - Soft Exudates

The dictionary follow the same structure as the DDR_EX_dataset dictionary structure.
"""
FGADR_EX_dataset = {
"x_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/train/images",
"y_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/EX_Masks/train/EX",

"x_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/val/images",
"y_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/EX_Masks/val/EX",

"x_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/test/images",
"y_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/EX_Masks/test/EX",
}

FGADR_MA_dataset = {
"x_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/train/images",
"y_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/MA_Masks/train/MA",

"x_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/val/images",
"y_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/MA_Masks/val/MA",

"x_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/test/images",
"y_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/MA_Masks/test/MA",
}

FGADR_HE_dataset = {
"x_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/train/images",
"y_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/HE_Masks/train/HE",

"x_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/val/images",
"y_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/HE_Masks/val/HE",

"x_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/test/images",
"y_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/HE_Masks/test/HE",
}

FGADR_SE_dataset = {
"x_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/train/images",
"y_train_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/SE_Masks/train/SE",

"x_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/val/images",
"y_valid_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/SE_Masks/val/SE",

"x_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/Images/test/images",
"y_test_dir": "/content/drive/MyDrive/Datasets/FGADR/Output/SE_Masks/test/SE",
}




##############################################################################
"""
DDR Models
The following dictionaries contains the paths to the models that are fitted to the DDR dataset.

The following dictionary structures are as follows:
  NAME = name of the model
  LOGGER_NAME = name of the logger that is used to log the model training and validation metrics
  CKPT_NAME = name of the checkpoint that is used to save the model
  CKPT_PATH = path to the checkpoint that is used to save the model
  LAST_CKPT_DIR = path to the last checkpoint that is used to load the model
  LAST_CKPT = path to the last checkpoint that is used to load the model

If the models are used with the spefied modifications outlied in the resreach project 
please add the following arguments into the dictionary structure:
  ARCH = name of the basline architecture that is used
  SMP_MODEL_NAME = name of the model that is used for modifying the encoder path of the basline architecture
  PRETRAINED = if True the model will use pre-trained weights
  ATTENTION = set it to "scse" so that the model will be modified with the attention mechanism called
  Special and Channel "Squeeze and Excitation" (scSE) block

"""

# DeepLab-V3+
deeplab = {
  "NAME": "deeplab",
  "LOGGER_DIR": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\deeplabv3",
  "CKPT_NAME": "deeplabv3-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\deeplabv3\weights",
  "LAST_CKPT_DIR": None,
  "LAST_CKPT": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\deeplabv3\weights2\last.ckpt"
}

# Unet++
u_netplusplus = {
  "NAME": "u_netplusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_netplusplus",
  "CKPT_NAME": "u_netplusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/models/u_netplusplus/weights",
  "LAST_CKPT_DIR": None
}

transfer_attention_u_effnet_plusplus= {
  "NAME": "transfer_attention_u_effnet_plusplus",
  "ARCH":'unetplusplus',
  "SMP_MODEL_NAME":'efficientnet-b4',
  "PRETRAINED": 'imagenet',
  "ATTENTION": 'scse',
  "LOGGER_DIR": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\transfer_attention_u_effnet_plusplus",
  "CKPT_NAME": "transfer_attention_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\transfer_attention_u_effnet_plusplus\weights",
  "LAST_CKPT_DIR": None,
  "LAST_CKPT": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\transfer_attention_u_effnet_plusplus\weights2\last.ckpt"
}

v2_transfer_attention_u_effnet_plusplus= {
  "NAME": "v2_transfer_attention_u_effnet_plusplus",
  "ARCH":'unetplusplus',
  "SMP_MODEL_NAME":'tu-tf_efficientnetv2_s_in21ft1k',
  "PRETRAINED": 'imagenet',
  "ATTENTION": 'scse',
  "LOGGER_DIR": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\V2_Transfer_Attention_U_Effnet_PlusPlus",
  "CKPT_NAME": "v2_transfer_attention_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\V2_Transfer_Attention_U_Effnet_PlusPlus\weights",
  "LAST_CKPT_DIR": None,
  "LAST_CKPT": r"C:\Users\Peter\Desktop\CS4525 Honours Project\DEV\src\1_Segmentation\models\V2_Transfer_Attention_U_Effnet_PlusPlus\last.ckpt"
}


# Unet
u_net = {
  "NAME": "u_net",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_net",
  "CKPT_NAME": "u_net-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/models/u_net/weights",
  "LAST_CKPT_DIR": None
}


# Efficientnet Unet
u_effnet = {
  "NAME": "u_effnet",
  "ARCH":'unet',
  "SMP_MODEL_NAME":'efficientnet-b4',
  "PRETRAINED": None,
  "ATTENTION": None,
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_effnet",
  "CKPT_NAME": "u_effnet-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/models/u_effnet/weights",
  "LAST_CKPT_DIR": None
}

# Efficientnet Unet v2
u_effnetv2 = {
  "NAME": "u_effnetv2",
  "ARCH":'unet',
  "SMP_MODEL_NAME":'tu-tf_efficientnetv2_m',
  "PRETRAINED": None,
  "ATTENTION": None,
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_effnetV2",
  "CKPT_NAME": "u_effnetv2-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_effnetV2/weights",
  "LAST_CKPT_DIR": None,
}

# EfficientNet Unet++ V2
u_effnet_plusplus = {
  "NAME": "u_effnet_plusplus",
  "ARCH":'unetplusplus',
  "SMP_MODEL_NAME":'efficientnet-b4',
  "PRETRAINED": None,
  "ATTENTION": None,
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/u_effnet_plusplus",
  "CKPT_NAME": "u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/models/u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}

# EfficientNet Unet++ with the Attention Mechanism
attention_u_effnet_plusplus= {
  "NAME": "attention_u_effnet_plusplus",
  "ARCH":'unetplusplus',
  "SMP_MODEL_NAME":'efficientnet-b4',
  "PRETRAINED": None,
  "ATTENTION": 'scse',
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/models/attention_u_effnet_plusplus",
  "CKPT_NAME": "attention_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/models/attention_u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}



################################################################################
"""
  FGADR Models
  The following are all the different models and their respective dictionaries that have been fitted to the FGADR dataset.

  DICE LOSS
"""

# DenseUnet for Hard Exudates
dense_unet_EX = {
  "NAME": "dense_unet_EX",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_net_EX",
  "CKPT_NAME": "dense_unet_EX-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_net_EX/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Microaneurysms
dense_unet_MA = {
  "NAME": "dense_unet_MA",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_net_MA",
  "CKPT_NAME": "dense_net_MA-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_net_MA/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Hemorrhages
dense_unet_HE = {
  "NAME": "dense_unet_HE",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_unet_HE",
  "CKPT_NAME": "dense_unet_HE-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_unet_HE/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Soft Exudates
dense_unet_SE = {
  "NAME": "dense_unet_SE",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_unet_SE",
  "CKPT_NAME": "dense_unet_SE-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/dense_unet_SE/weights",
  "LAST_CKPT_DIR": None
}

################################################################################
# EfficentNet Unet++ for Hard Exudates
EX_u_effnet_plusplus= {
  "NAME": "EX_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/EX_u_effnet_plusplus",
  "CKPT_NAME": "EX_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/EX_u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Microaneurysms
MA_u_effnet_plusplus= {
  "NAME": "MA_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/MA_u_effnet_plusplus",
  "CKPT_NAME": "MA_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/MA_u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Hemorrhages
HE_u_effnet_plusplus= {
  "NAME": "HE_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/HE_u_effnet_plusplus",
  "CKPT_NAME": "HE_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/HE_u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Soft Exudates
SE_u_effnet_plusplus= {
  "NAME": "SE_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/SE_u_effnet_plusplus",
  "CKPT_NAME": "SE_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/SE_u_effnet_plusplus/weights",
  "LAST_CKPT_DIR": None
}

######################################################################################
######################################################################################

"""
CROSS ENTROPY

"""
# FGADR Dense Unet
# DenseUnet for Hard Exudates
CE_dense_unet_EX = {
  "NAME": "dense_unet_EX",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/EX",
  "CKPT_NAME": "dense_unet_EX-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/EX/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Microaneurysms
CE_dense_unet_MA = {
  "NAME": "dense_unet_MA",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/MA",
  "CKPT_NAME": "dense_net_MA-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/MA/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Hemorrhages
CE_dense_unet_HE = {
  "NAME": "dense_unet_HE",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/HE",
  "CKPT_NAME": "dense_unet_HE-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/HE/weights",
  "LAST_CKPT_DIR": None
}

# DenseUnet for Soft Exudates
CE_dense_unet_SE = {
  "NAME": "dense_unet_SE",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/SE",
  "CKPT_NAME": "dense_unet_SE-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_densenet/SE/weights",
  "LAST_CKPT_DIR": None
}

################################################################################
# EfficentNet Unet++ for Hard Exudates
CE_EX_u_effnet_plusplus= {
  "NAME": "EX_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/EX",
  "CKPT_NAME": "EX_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/EX/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Macroaneurysms
CE_MA_u_effnet_plusplus= {
  "NAME": "MA_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/MA",
  "CKPT_NAME": "MA_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/MA/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Hemorrhages
CE_HE_u_effnet_plusplus= {
  "NAME": "HE_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/HE",
  "CKPT_NAME": "HE_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/HE/weights",
  "LAST_CKPT_DIR": None
}

# EfficentNet Unet++ for Soft Exudates
CE_SE_u_effnet_plusplus= {
  "NAME": "SE_u_effnet_plusplus",
  "LOGGER_DIR": "/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/SE",
  "CKPT_NAME": "SE_u_effnet_plusplus-{epoch:02d}-{val_loss:.2f}",
  "CKPT_DIR":"/content/drive/MyDrive/src/1_Segmentation/FGADR_Models/CE_ueffnet/SE/weights",
  "LAST_CKPT_DIR": None
}