import onnx
import onnxruntime as ort
import numpy as np

ONNX_PATH = "onnx/SiliconSight.onnx"

def verify():
    print(f"🧐 Checking {ONNX_PATH}...")

    # 1. Check Model Structure
    try:
        model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(model)
        print("   ✅ Valid ONNX Format!")
    except Exception as e:
        print(f"   ❌ Corrupted File: {e}")
        return

    # 2. Test Inference (Run a fake image through it)
    try:
        session = ort.InferenceSession(ONNX_PATH)
        
        # Create dummy input (1 image, 3 channels, 224x224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run model
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"   ✅ Inference Successful!")
        print(f"   Output Shape: {outputs[0].shape} (Should be [1, 7])")
        
    except Exception as e:
        print(f"   ❌ Runtime Error: {e}")

if __name__ == "__main__":
    verify()