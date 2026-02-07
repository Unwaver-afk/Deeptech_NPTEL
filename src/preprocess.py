from torchvision import transforms
from transformers import AutoImageProcessor

def get_transforms(model_name="facebook/dinov2-base"):
    """
    Returns the standard image normalization and resizing transforms 
    required for the DinoV2 model.
    """
    # Load processor to get exact mean and std used during pre-training
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])