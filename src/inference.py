import torch
import os
from PIL import Image
from transformers import Dinov2ForImageClassification, AutoImageProcessor
from torchvision import transforms

# Configuration
MODEL_PATH = "hackathon_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Class Mapping 
# 0 = Clean, 1-6 = Defects
ID2LABEL = {
    0: 'Clean',
    1: 'Bridge',
    2: 'Opens',
    3: 'Cracks',
    4: 'CMP Residue',
    5: 'Particles',
    6: 'Scratch'
}

class DefectPredictor:
    def __init__(self, model_dir=MODEL_PATH):
        """
        Initializes the model and prepares it for inference.
        """
        print(f"Loading model from {model_dir} on {DEVICE}...")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model folder '{model_dir}' not found. Did you run train.py?")

        try:
            # Load Model
            self.model = Dinov2ForImageClassification.from_pretrained(model_dir)
            self.model.to(DEVICE)
            self.model.eval()
            
            # Load Preprocessing Stats (Mean/Std) from base model
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            
            # Define Transform (Same as training)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
            ])
            
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_path):
        """
        Predicts the class of a single image.
        Args:
            image_path (str): Path to the image file.
        Returns:
            dict: {'label': str, 'confidence': float, 'inference_time': float}
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}

        try:
            # 1. Open and Convert Image
            image = Image.open(image_path).convert("RGB")
            
            # 2. Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # 3. Inference
            with torch.no_grad():
                outputs = self.model(pixel_values=input_tensor)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Get Top Prediction
                confidence, pred_id = torch.max(probs, dim=1)
                pred_id = pred_id.item()
                confidence = confidence.item()
                
                label = ID2LABEL.get(pred_id, "Unknown")
                
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {ID2LABEL[i]: probs[0][i].item() for i in range(len(ID2LABEL))}
            }

        except Exception as e:
            return {"error": str(e)}

# --- Quick Test Block ---
if __name__ == "__main__":
    # This block only runs if you execute 'python src/inference.py' directly
    import glob
    import random
    
    print("--- Running Self-Test ---")
    try:
        predictor = DefectPredictor()
        
        # Grab a random test image to verify it works
        test_images = glob.glob("data/test/*/*.jpg")
        
        if test_images:
            random_img = random.choice(test_images)
            print(f"\nTesting on image: {random_img}")
            
            result = predictor.predict(random_img)
            
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
        else:
            print("No test images found to run automatic check.")
            
    except Exception as e:
        print(f"Self-test failed: {e}")