import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SemiconductorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom Dataset for loading Semiconductor images.
        
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of integer labels (0=Clean, 1-6=Defects).
            transform (callable, optional): Transform to apply to the image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Ensure image is RGB (3 channels) as required by the model
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images to prevent training crash
            # Returns a black image of standard size
            print(f"Warning: Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)