""" 
This Class has been adapted from the following github repository: 
https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
"""

import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *
import pytorch_lightning as pl

# Added by Dewei
import torch.nn.functional as F

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class SegmentationModel(pl.LightningModule):
    """
    This class creates the neural network model by also defining the loss function and the optimizer
    for the task of Lesion Segmentation.
    Addionally, this class provides additional functions that are used to train, validate and test the model.
    Futhermore, this class has combined the aforementioned functions into one function called `shared_step` and `shared_epoch_end
    by also calculating the diffrent performance metrics.
    
    Args:
        arch (str): The name of the model architecture.
        encoder_name (str): The name of the encoder.
        in_channels (int): The number of input channels.
        out_classes (int): The number of output classes.
        attention_type (str): The type of attention mechanism.
        lr (float): The learning rate.
        pretrained (str): If True, the model will use pretrained weights.
        **kwargs: Additional arguments for the model.

    Returns:
        model (nn.Module): The neural network model based on the Pytorch nn class.
    """


    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, attention_type=None, lr = 1e-3, pretrained=None, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=out_classes, 
            decoder_attention_type=attention_type, 
            **kwargs
        )
        self.lr = lr
        self.save_hyperparameters()
        
        # Preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Here is should be defined which Loss Function should be used
        # In this case Dice Loss has been defined and takes raw logits as input
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        """
        This method will forward a given image and normalize it by subtracting mean and dividing by the standard deviation.
        One the normalization is finished, the image will be passed to the model for predicting the segmentation mask.

        Args:
            image (torch.Tensor): The image to be forwarded.
        Returns:
            mask (torch.Tensor): The predicted mask.
        """
        # normalize image here
        image = (image - self.mean) / self.std
        # predict the segmentation mask
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """
        This method will propagate the given batch of images and masks through the model.
        The model will predict the segmentation mask with the forward method and the loss will be calculated.
        Addionally, the TP, FP, FN and TN metrics will be calculated.
        """
        
        image, _= batch
       
        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4

        # Checking if the image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        _, mask = batch

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values is in-between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        # Forward pass
        logits_mask = self.forward(image)
        
        # The loss is calculated by the Dice Loss function
        m = nn.Sigmoid()
        # Binary Cross Entropy
        # loss = nn.BCELoss()(m(logits_mask.squeeze()), mask.squeeze())
        # DiceLoss
        # loss = DiceLoss()(m(logits_mask.squeeze()), mask.squeeze())
        # BCE + Dice loss
        # mylambda = 1/2
        # loss = mylambda*(nn.BCELoss()(m(logits_mask.squeeze()), mask.squeeze())) + (1-mylambda)*DiceLoss()(m(logits_mask.squeeze()), mask.squeeze())
        # BCE logit + Dice loss
        #loss = (nn.BCEWithLogitsLoss()(m(logits_mask.squeeze()), mask.squeeze()) + DiceLoss()(m(logits_mask.squeeze()), mask.squeeze()))/2
        # Binary Cross Entropy with Logit
        loss = nn.BCEWithLogitsLoss()(m(logits_mask.squeeze()), mask.squeeze())

        # computing the TP, FP, FN and TN metrics 
        # by applying the sigmoid operation
        # for a thresholding set at 0.5 
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # for a thresholding set at 0.25
        #pred_mask = (prob_mask > 0.25).float()

        # Computing the TP, FP, FN and TN metrics    
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """
        This method will calculate the average loss and the average metrics.
        These calculations will be done at the end of every epoch.
        The calculated metrics will be saved in the log file 
        and logged through the training, validaition and testing processes.


        Args:
            outputs (dict): The outputs of the model.
            stage (str): The stage of the model.

        """
        loss = torch.stack([x["loss"] for x in outputs])
        mean_loss = torch.mean(loss[-100:])
        
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        # Calculate the different metrics such as IoU, Dice Score, Precision, Recall, Accuracy.
        # The reduction parameter is set to "micro" which computes a global average  by counting 
        # the sums of the True Positives (TP), False Negatives (FN), and False Positives (FP).
        # For binary segmentation the "micro" is equivalent to "macro" reduction 
        # which treats all classes equally regardless of their support values.
        mean_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro", zero_division=1e-20)
        mean_F1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro", zero_division=1e-20)
        mean_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, zero_division=1e-20)
        mean_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro", zero_division=1e-20)
        mean_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro", zero_division=1e-20)
        
        # aggregate step metics
        metrics = {
            f"{stage}_loss": mean_loss,
            f"{stage}_mean_iou": mean_iou,
            f"{stage}_mean_F1": mean_F1,
            f"{stage}_mean_accuracy": mean_accuracy,
            f"{stage}_mean_precision": mean_precision,
            f"{stage}_mean_recall": mean_recall,
        }
        # log the metrics
        self.log_dict(metrics, prog_bar=True)
        

    def training_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and metrics 
        and it substitutes the traditional training step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and metrics.
        """
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional training epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and metrics 
        and it substitutes the traditional validation step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and TP, FP, FN and TN metrics.
        """
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional validation epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        """
        This method will be called at every batch.
        It will call the shared_step method to calculate the loss and metrics 
        and it substitutes the traditional testing step 

        Args: 
            batch (tuple): The batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: The loss and TP, FP, FN and TN metrics.
        """
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        """
        This method will be called at the end of every epoch.
        It will call the shared_epoch_end method to calculate the average loss and metrics.
        It substitutes the traditional testing epoch end.

        Args:
            outputs (dict): The outputs of the model.
        
        Returns:
            dict: The average loss and metrics.
        """
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """
        This method will be called to configure the optimizers and the learning rate scheduler.

        Returns:
            list: optimizer and learning rate scheduler
        """
        # define the optimizer
        # here we have applied the ADAM optimizer
        #optimizer = optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)


        # define the learning rate scheduler
        # here we have applied the ReduceLROnPlateau scheduler
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                                 mode='min', 
                                                                                 patience=6,
                                                                                 min_lr=8.0e-05), 
                         "monitor": "train_loss"}
        return [optimizer], [lr_schedulers]


