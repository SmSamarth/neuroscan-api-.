from fastapi import FastAPI, File, UploadFile
from preprocessing.opencv_pipeline import MRIProcessor
import tensorflow as tf
import numpy as np

# ==========================================
# 1. THE CLASSIFIER CLASS
# (Moved here so you don't need a separate file)
# ==========================================
class MRIClassifier:
    def __init__(self, model_path, labels):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = labels

    def predict(self, processed_image):
        # Ensure dimensions are (1, height, width, channels)
        if processed_image.ndim == 3:
            img_array = np.expand_dims(processed_image, axis=0)
        else:
            img_array = processed_image
            
        predictions = self.model.predict(img_array)
        probabilities = predictions[0] # Use raw output assuming Softmax is in the model
        
        class_index = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            "label": self.labels[class_index],
            "confidence_score": confidence
        }

# ==========================================
# 2. APP SETUP
# ==========================================
app = FastAPI(title="NeuroScan ML Service")
processor = MRIProcessor()

# ==========================================
# 3. ALZHEIMER's ENDPOINT
# ==========================================
ALZ_LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
ALZ_MODEL_PATH = r"C:\Users\samar\OneDrive\Desktop\NeuroScan\neuroscan\ml-service\models\trained_alzheimers_model.keras"

alzheimer_classifier = MRIClassifier(ALZ_MODEL_PATH, ALZ_LABELS)

@app.post("/predict/alzheimer")
async def predict_alzheimer(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        processed_image = processor.preprocess(image_bytes)
        ai_result = alzheimer_classifier.predict(processed_image)
        
        return {
            "status": "success",
            "diagnosis": ai_result["label"], 
            "confidence": round(ai_result["confidence_score"] * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 4. BRAIN TUMOR ENDPOINT
# ==========================================
TUMOR_LABELS = ["no", "yes"]
TUMOR_MODEL_PATH = r"C:\Users\samar\OneDrive\Desktop\NeuroScan\neuroscan\ml-service\models\trained_tumor_model.keras"

tumor_classifier = MRIClassifier(TUMOR_MODEL_PATH, TUMOR_LABELS)

@app.post("/predict/tumor")
async def predict_tumor(file: UploadFile = File(...)):
    image_bytes = await file.read()

@app.post("/predict/comprehensive")
async def predict_comprehensive(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        # Preprocess the image once
        processed_image = processor.preprocess(image_bytes)
        
        # Run the image through BOTH AI models
        alz_result = alzheimer_classifier.predict(processed_image)
        tumor_result = tumor_classifier.predict(processed_image)
        
        # Combine the results into a single patient report
        return {
            "status": "success",
            "comprehensive_report": {
                "alzheimers_screening": {
                    "finding": alz_result["label"],
                    "confidence": f"{round(alz_result['confidence_score'] * 100, 2)}%"
                },
                "tumor_screening": {
                    "finding": tumor_result["label"],
                    "confidence": f"{round(tumor_result['confidence_score'] * 100, 2)}%"
                }
            }
        }
    except Exception as e:
        return {"error": str(e)}
    try:
        processed_image = processor.preprocess(image_bytes)
        ai_result = tumor_classifier.predict(processed_image)
        
        return {
            "status": "success",
            "diagnosis": ai_result["label"], 
            "confidence": round(ai_result["confidence_score"] * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}