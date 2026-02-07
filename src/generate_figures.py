import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from matplotlib.colors import ListedColormap

# Configuration
DATA_ROOT = "data"
FIGURES_DIR = "figures"

# Create output directory
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)
    print(f"Created '{FIGURES_DIR}' directory.")

# Load Data Statistics
print(f"Scanning '{DATA_ROOT}' for statistics...")

if not os.path.exists(DATA_ROOT):
    print(f"Error: '{DATA_ROOT}' folder not found.")
    exit()

stats = {
    "clean": 0,
    "defect": 0,
    "train": 0,
    "val": 0,
    "test": 0
}

sample_defect_path = None

# Count files in train/val/test splits
for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATA_ROOT, split)
    if not os.path.exists(split_path): continue
    
    # Count Clean images
    clean_imgs = glob.glob(os.path.join(split_path, "clean", "*.*"))
    count_clean = len(clean_imgs)
    stats["clean"] += count_clean
    stats[split] += count_clean
    
    # Count Defect images
    other_path = os.path.join(split_path, "other")
    if os.path.exists(other_path):
        for defect_folder in os.listdir(other_path):
            d_path = os.path.join(other_path, defect_folder)
            if os.path.isdir(d_path):
                d_imgs = glob.glob(os.path.join(d_path, "*.*"))
                count_defect = len(d_imgs)
                stats["defect"] += count_defect
                stats[split] += count_defect
                
                # Capture sample image for visualization
                if sample_defect_path is None and len(d_imgs) > 0:
                    sample_defect_path = d_imgs[0]

print(f"Statistics Loaded: {stats}")

# Helper to load sample image
def get_sample_image():
    if sample_defect_path:
        img = Image.open(sample_defect_path).convert("L") 
        return np.array(img)
    else:
        return np.random.rand(224, 224)

example_wafer = get_sample_image()

# Plot 1: Human vs Machine View
def plot_human_vs_machine():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Operator View
    axes[0].imshow(example_wafer, cmap='gray') 
    axes[0].set_title("Operator View (UI)\nOptimized for Human Comfort", fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Model View (High Contrast)
    axes[1].imshow(example_wafer, cmap='inferno') 
    axes[1].set_title("Model Input (ViT)\nOptimized for Signal Contrast", fontsize=12, weight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "human_vs_machine.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# Plot 2: Preprocessing Pipeline
def plot_pipeline():
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.text(0.5, 0.5, "Raw File\n(.jpg / .png)", 
             ha='center', va='center', fontsize=14, fontfamily='monospace', weight='bold')
    ax1.set_title("1. Raw Data Input", weight='bold')
    ax1.axis('off')
    
    ax_arrow = fig.add_subplot(1, 3, 2)
    ax_arrow.text(0.5, 0.5, "Resize (224x224)\nNormalize ->", ha='center', va='center', fontsize=12, weight='bold')
    ax_arrow.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(example_wafer, cmap='inferno', interpolation='nearest')
    ax3.set_title("3. Vision Tensor", weight='bold')
    ax3.axis('off')
    
    save_path = os.path.join(FIGURES_DIR, "preprocessing_pipeline.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# Plot 3: Class Distribution
def plot_distribution():
    clean = stats["clean"]
    defects = stats["defect"]
    total = clean + defects
    
    labels = ['Clean Wafers', 'Defected Wafers']
    counts = [clean, defects]
    colors = ['#8cd2d2', '#c62828']
    
    plt.figure(figsize=(7, 7))
    if total > 0:
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
                textprops={'fontsize': 12, 'weight': 'bold'}, shadow=True)
        plt.title(f"Dataset Class Balance\n(Total: {total:,})", fontsize=14, weight='bold')
    else:
        plt.text(0.5, 0.5, "Dataset Empty", ha='center')
        plt.title("Dataset Empty")
    
    save_path = os.path.join(FIGURES_DIR, "dataset_balance.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# Plot 4: Data Split
def plot_training_split():
    train_size = stats["train"]
    val_size = stats["val"]
    test_size = stats["test"]
    total = train_size + val_size + test_size
    
    if total == 0:
        print("Cannot plot data split: No images found.")
        return

    sizes = [train_size, val_size, test_size]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['#2e7d32', '#fdd835', '#1565c0']
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=(8, 8))
    
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%\n({v:,})'.format(p=pct, v=val)
        return my_autopct

    plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), 
            startangle=90, colors=colors, explode=explode, 
            textprops={'fontsize': 12, 'weight': 'bold'}, shadow=True)
    
    plt.title(f"Experimental Data Split\nTotal: {total:,} images", fontsize=14, weight='bold')
    
    save_path = os.path.join(FIGURES_DIR, "data_split.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    print("\nGenerating Project Figures...")
    print("="*40)
    
    plot_human_vs_machine()
    plot_pipeline()
    plot_distribution()
    plot_training_split()
    
    print("="*40)
    print(f"Success! All figures saved to: {os.path.abspath(FIGURES_DIR)}")