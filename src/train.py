import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Dinov2ForImageClassification
from torch.optim import AdamW

# Local imports
try:
    from src.dataset import SemiconductorDataset
    from src.preprocess import get_transforms
except ImportError:
    from dataset import SemiconductorDataset
    from preprocess import get_transforms

# Configuration
DATA_ROOT = "data"
SAVE_PATH = "hackathon_model"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Class definition (0 is Clean, 1-6 are Defects)
CLASSES = ['clean', 'bridge', 'opens', 'cracks', 'cmp_residue', 'particles', 'scratch']
NUM_CLASSES = len(CLASSES)

def load_data_from_split(split_name):
    """
    Scans the specific split folder (train/val) to gather image paths and labels.
    Structure expected:
      - data/{split}/clean
      - data/{split}/other/{defect_class}
    """
    split_dir = os.path.join(DATA_ROOT, split_name)
    paths = []
    labels = []

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # 1. Load Clean Images (Label 0)
    clean_path = os.path.join(split_dir, "clean")
    if os.path.exists(clean_path):
        clean_imgs = glob.glob(os.path.join(clean_path, "*.*"))
        # Filter for valid image extensions
        clean_imgs = [x for x in clean_imgs if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        paths.extend(clean_imgs)
        labels.extend([0] * len(clean_imgs))

    # 2. Load Defect Images (Labels 1-6)
    other_dir = os.path.join(split_dir, "other")
    if os.path.exists(other_dir):
        # We start from index 1 because 0 is 'clean'
        for i, class_name in enumerate(CLASSES[1:], start=1):
            defect_path = os.path.join(other_dir, class_name)
            if os.path.exists(defect_path):
                defect_imgs = glob.glob(os.path.join(defect_path, "*.*"))
                defect_imgs = [x for x in defect_imgs if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                paths.extend(defect_imgs)
                labels.extend([i] * len(defect_imgs))
    
    return paths, labels

def main():
    print(f"Initializing training on {DEVICE}...")

    # Load file paths
    train_paths, train_labels = load_data_from_split("train")
    val_paths, val_labels = load_data_from_split("val")

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    if len(train_paths) == 0:
        print("Error: No training data found. Check data directory structure.")
        return

    # Dataset and DataLoader
    transform = get_transforms()
    
    train_dataset = SemiconductorDataset(train_paths, train_labels, transform=transform)
    val_dataset = SemiconductorDataset(val_paths, val_labels, transform=transform)

    # FIX: num_workers=0 prevents multiprocessing crashes on Mac
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model Setup
    id2label = {str(i): c for i, c in enumerate(CLASSES)}
    label2id = {c: str(i) for i, c in enumerate(CLASSES)}

    model = Dinov2ForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )
    model.to(DEVICE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            outputs = model(pixel_values=images, labels=targets)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(pixel_values=images, labels=targets)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Calculate accuracy
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == targets).sum().item()
                total_samples += targets.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples

        print(f"\nEpoch {epoch+1} complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # Save model
    print(f"Saving model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    print("Training finished successfully.")

if __name__ == "__main__":
    main()