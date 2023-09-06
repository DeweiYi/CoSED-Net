import albumentations as albu

def get_training_augmentation():
    """
    1. Resizes the images and masks for the test set
    with the dimensions of height=512 and width=512
    2. Center Crops the image 
    3. Applies the CLAHE (Contrast Limited Adaptive 
    Histogram Equalization) method 

    Return:
        transform: albumentations.Compose
    """
    train_transform = [

        albu.Resize(height=512, width=512),
        albu.CenterCrop(height=512, width=512),
        albu.CLAHE(p=1),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """
    Resizes the images and masks for both validation and test sets
    with the dimensions of height=512 and width=512

    Return:
        transform: albumentations.Compose
    """
    test_transform = [
        albu.Resize(height=512, width=512),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Converts the image and mask to tensors with float type
    Converts the shapes from HWC ==> CHW (Channels, Height, Weight)

    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing():
    """Construct preprocessing transform that is received
    from the preprocessesing that comes with different
    pretrained neural networks
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        # albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)