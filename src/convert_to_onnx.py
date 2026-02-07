import torch
import torch.onnx
import os
from transformers import Dinov2ForImageClassification

# --- CONFIGURATION ---
MODEL_PATH = "hackathon_model"  
ONNX_PATH = "SiliconSight.onnx"       
OPSET_VERSION = 12              

def export_to_onnx():
    print(f"Starting ONNX Export...")
    
    # 1. Load the Trained Model
    print(f"   Loading model from '{MODEL_PATH}'...")
    try:
        model = Dinov2ForImageClassification.from_pretrained(MODEL_PATH)
        model.eval() 
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"    Error: {e}")
        print("   Did you run train.py first?")
        return

    # 2. Create Dummy Input
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # 3. Export
    print(f"   📦 Exporting to '{ONNX_PATH}'...")
    try:
        torch.onnx.export(
            model,                      # The model
            dummy_input,                # Dummy input
            ONNX_PATH,                  # Output file
            export_params=True,         # Store the trained weights inside the file
            opset_version=OPSET_VERSION, 
            do_constant_folding=True,   # Optimization: Pre-calculate constant values
            input_names=['input'],      # Name of the input layer
            output_names=['output'],    # Name of the output layer
            dynamic_axes={              # Allow variable batch sizes 
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"   🎉 SUCCESS! Model saved to {os.path.abspath(ONNX_PATH)}")
        print(f"   Filesize: {os.path.getsize(ONNX_PATH) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"   Export Failed: {e}")

if __name__ == "__main__":
    export_to_onnx()