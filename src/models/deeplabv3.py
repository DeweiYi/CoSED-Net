import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *
import pytorch_lightning as pl

class DeepLabV3Plus(pl.LightningModule):

    """
    This class creates the neural network model called DeepLabV3 by also defining the loss function and the optimizer
    for the task of Lesion Segmentation.
    Addionally, this class provides additional functions that are used to train, validate and test the model.
    Futhermore, this class has combined the aforementioned functions into one function called `shared_step` and `shared_epoch_end
    by also calculating the diffrent performance metrics.
    
    Args:
        model_name: The name of the model that is used for training.
        encoder_weights: The weights that are used for training the model.
        in_channels: The number of channels that are used for training the model.
        classes: The number of classes that are used for training the model.
        lr: The learning rate that is used for training the model.
        attention_type: The type of attention that is used for training the model.
        pretrained: The weights that are used for training the model.

    Returns:
        model (nn.Module): The neural network model based on the Pytorch nn class.

    For more in-depth documentation of this class please see the model_creator.py file in the Models folder.
    """

    def __init__(self, in_channels=3, out_classes=1, lr = 1e-3, attention_type=None, pretrained=None):
        super().__init__()
        self.model = smp.create_model(
            "DeepLabV3Plus", 
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=out_classes,
        )
        self.lr = lr
        self.save_hyperparameters()

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params("tu-tf_efficientnetv2_m")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # loss function
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

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
        Additional assertion for the shapes of the images and masks is performed.

        For more in-depth documentation of this class please see the model_creator.py file in the Models folder.
        """
        image, _= batch
       
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        _, mask = batch
        assert mask.ndim == 4


        logits_mask = self.forward(image)
        
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
      
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
        
        
        mean_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        mean_F1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        mean_accuracy = smp.metrics.accuracy(tp, fp, fn, tn)
        mean_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        mean_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        
        # aggregate step metics
        metrics = {
            f"{stage}_loss": mean_loss,
            f"{stage}_mean_iou": mean_iou,
            f"{stage}_mean_F1": mean_F1,
            f"{stage}_mean_accuracy": mean_accuracy,
            f"{stage}_mean_precision": mean_precision,
            f"{stage}_mean_recall": mean_recall,
        }
        
        self.log_dict(metrics, prog_bar=True)
        

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                                 mode='min', 
                                                                                 patience=1,
                                                                                 min_lr=8.0e-05), 
                         "monitor": "train_loss"}
        return [optimizer], [lr_schedulers]