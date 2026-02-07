import torch
from transformers import Dinov2ForImageClassification

def load_model(model_path=None, device="cpu"):
    if model_path:
        # Load trained weights
        model = Dinov2ForImageClassification.from_pretrained(model_path)
    else:
        # Load fresh pre-trained backbone
        model = Dinov2ForImageClassification.from_pretrained(
            "facebook/dinov2-base",
            num_labels=2
        )
    
    model.to(device)
    return model