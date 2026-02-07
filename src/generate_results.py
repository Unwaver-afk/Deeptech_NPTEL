import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import glob
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    classification_report
)
from sklearn.preprocessing import label_binarize
from transformers import Dinov2ForImageClassification, AutoImageProcessor
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configuration
MODEL_PATH = "./hackathon_model"
DATA_ROOT = "data"
TEST_SPLIT = "test"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32

# Class definitions (0=Clean, 1-6=Defects)
CLASSES = ['clean', 'bridge', 'opens', 'cracks', 'cmp_residue', 'particles', 'scratch']
NUM_CLASSES = len(CLASSES)

print(f"Initializing Evaluation on {DEVICE}...")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 1. Load Model
print("Loading Model...")
try:
    id2label = {str(i): c for i, c in enumerate(CLASSES)}
    label2id = {c: str(i) for i, c in enumerate(CLASSES)}
    
    model = Dinov2ForImageClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)
    model.eval()
    
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
except Exception as e:
    print(f"Model Error: {e}")
    print("Ensure you have trained the model first.")
    exit()

# 2. Prepare Data (Scanning Test Folder)
print("Scanning Test Data...")
all_image_paths = []
all_image_labels = []

test_dir = os.path.join(DATA_ROOT, TEST_SPLIT)

# Scan Clean Images (Class 0)
clean_path = os.path.join(test_dir, "clean")
if os.path.exists(clean_path):
    images = glob.glob(os.path.join(clean_path, "*.*"))
    images = [x for x in images if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    all_image_paths.extend(images)
    all_image_labels.extend([0] * len(images))

# Scan Defect Images (Classes 1-6)
other_dir = os.path.join(test_dir, "other")
if os.path.exists(other_dir):
    # Skip 'clean' at index 0, iterate defects starting at index 1
    for i, class_name in enumerate(CLASSES[1:], start=1):
        folder_path = os.path.join(other_dir, class_name)
        if os.path.exists(folder_path):
            images = glob.glob(os.path.join(folder_path, "*.*"))
            images = [x for x in images if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            all_image_paths.extend(images)
            all_image_labels.extend([i] * len(images))

print(f"Found {len(all_image_paths)} total images for evaluation.")

if len(all_image_paths) == 0:
    print("Error: No images found in data/test. Check your folder structure.")
    exit()

# 3. Custom Dataset
class EvalDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
            
        return self.transform(img), label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

dataset = EvalDataset(all_image_paths, all_image_labels, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- PHASE 1: INFERENCE ---
print("Running Inference & Benchmarking Speed...")
y_true = []
y_pred = []
y_probs = []

start_time = time.time()

with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        
        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        y_probs.extend(probs.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(lbls.numpy())

end_time = time.time()

# Calculate Latency
total_samples = len(y_true)
total_time = end_time - start_time
latency_ms = (total_time / total_samples) * 1000
fps = total_samples / total_time
print(f"Speed: {latency_ms:.2f} ms/image ({fps:.0f} FPS)")

# --- PHASE 2: GENERATE METRICS ---

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"\nAccuracy:  {acc:.2%}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"MCC:       {mcc:.4f}")

# Save JSON
metrics_dict = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "mcc": mcc,
    "latency_ms": latency_ms,
    "fps": fps,
    "samples": total_samples,
    "classes": CLASSES
}
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)
print("Saved metrics.json")

# --- PHASE 3: PLOTTING ---

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Confusion Matrix (Acc={acc:.2%})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# 2. Multi-Class ROC Curve
try:
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    y_probs_np = np.array(y_probs)
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(CLASSES):
        # Only plot if class exists in the test set
        if i < y_true_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_np[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
    plt.close()
except Exception as e:
    print(f"Could not generate ROC: {e}")

# 3. Confidence Histogram
max_probs = np.max(y_probs, axis=1)
plt.figure(figsize=(8, 5))
sns.histplot(max_probs, bins=20, kde=True, color='purple')
plt.title('Model Confidence Distribution')
plt.xlabel('Confidence Score (0.0 - 1.0)')
plt.ylabel('Count')
plt.axvline(0.5, color='red', linestyle='--')
plt.savefig(os.path.join(RESULTS_DIR, "confidence_histogram.png"))
plt.close()

print(f"\nAll results saved to: {os.path.abspath(RESULTS_DIR)}")