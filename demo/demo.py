import os
import sys
import random
import shutil
import glob
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import DefectPredictor

# Configuration
TEST_DATA_DIR = "data/dataset/test"
ASSETS_DIR = os.path.join("demo", "assets")
CLEAN_DIR = os.path.join(ASSETS_DIR, "accepted")
DEFECT_DIR = os.path.join(ASSETS_DIR, "rejected")
REPORT_FILE = os.path.join(ASSETS_DIR, "inspection_report.html")

# ANSI Colors for Terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def setup_directories():
    if os.path.exists(ASSETS_DIR):
        shutil.rmtree(ASSETS_DIR)
    
    os.makedirs(CLEAN_DIR)
    os.makedirs(DEFECT_DIR)

def get_test_images(n):
    # Gather all image files
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(TEST_DATA_DIR, "**", ext), recursive=True))
    
    if not all_files:
        print("Error: No images found in data/test.")
        sys.exit(1)
        
    # Shuffle and pick n
    random.shuffle(all_files)
    return all_files[:n]

def generate_html_header():
    return """
    <html>
    <head>
        <title>Inspection Report</title>
        <style>
            body { font-family: sans-serif; background: #f4f4f9; padding: 20px; }
            h1 { color: #333; }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; }
            .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px; text-align: center; }
            img { width: 100%; border-radius: 4px; }
            .status { font-weight: bold; margin-top: 10px; display: block; }
            .clean { color: green; }
            .defect { color: red; }
        </style>
    </head>
    <body>
        <h1>Wafer Inspection Report</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <div class='grid'>
    """

def run_demo():
    print(f"{Colors.HEADER}--- SILICON SIGHT AUTOMATED INSPECTION ---{Colors.ENDC}")
    print("Initializing System...")
    
    setup_directories()
    
    try:
        predictor = DefectPredictor()
    except Exception as e:
        print(f"{Colors.FAIL}Model Error: {e}{Colors.ENDC}")
        return

    # User Input
    while True:
        try:
            val = input(f"\n{Colors.BOLD}Enter number of wafers to scan (e.g. 5): {Colors.ENDC}")
            num_wafers = int(val)
            if num_wafers > 0: break
        except ValueError:
            pass

    # Start Inspection
    samples = get_test_images(num_wafers)
    html_content = generate_html_header()
    
    print("-" * 65)
    print(f"{'TIMESTAMP':<10} | {'WAFER ID':<15} | {'PREDICTION':<12} | {'CONFIDENCE':<8} | {'STATUS'}")
    print("-" * 65)

    stats = {"ok": 0, "ng": 0}

    for i, img_path in enumerate(samples):
        # Simulate scanning delay
        time.sleep(0.4)
        
        # Predict
        result = predictor.predict(img_path)
        label = result['label']
        conf = result['confidence']
        filename = os.path.basename(img_path)
        
        # Logic
        is_clean = (label.lower() == 'clean')
        
        if is_clean:
            status_text = "ACCEPTED"
            color = Colors.OKGREEN
            stats['ok'] += 1
            dest = os.path.join(CLEAN_DIR, filename)
            html_class = "clean"
        else:
            status_text = "REJECTED"
            color = Colors.FAIL
            stats['ng'] += 1
            dest = os.path.join(DEFECT_DIR, filename)
            html_class = "defect"

        # Save File
        shutil.copy2(img_path, dest)

        # Print to Console
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp:<10} | {filename[:15]:<15} | {label:<12} | {conf:.1%}    | {color}{status_text}{Colors.ENDC}")

        # Add to HTML
        rel_path = os.path.relpath(dest, ASSETS_DIR)
        html_content += f"""
            <div class='card'>
                <img src='{rel_path}'>
                <span class='status {html_class}'>{status_text}</span>
                <br><small>{label} ({conf:.1%})</small>
            </div>
        """

    # Finalize
    html_content += "</div></body></html>"
    with open(REPORT_FILE, "w") as f:
        f.write(html_content)

    print("-" * 65)
    print(f"{Colors.BOLD}SCAN COMPLETE{Colors.ENDC}")
    print(f"Total:    {len(samples)}")
    print(f"Accepted: {Colors.OKGREEN}{stats['ok']}{Colors.ENDC}")
    print(f"Rejected: {Colors.FAIL}{stats['ng']}{Colors.ENDC}")
    print(f"\nVisual Report: {Colors.OKBLUE}{os.path.abspath(REPORT_FILE)}{Colors.ENDC}")

if __name__ == "__main__":
    run_demo()