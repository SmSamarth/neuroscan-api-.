import gc
from fastapi import FastAPI, File, UploadFile
from preprocessing.opencv_pipeline import MRIProcessor
import tensorflow as tf
import numpy as np

# ==========================================
# 1. THE CLASSIFIER CLASS (Memory Optimized)
# ==========================================
class MRIClassifier:
    def __init__(self, model_path, labels):
        self.model_path = model_path
        self.labels = labels
        self.model = None # Start with nothing in memory

    def load_model(self):
        """Loads the model into RAM only when needed."""
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

    def unload_model(self):
        """Deletes the model from RAM to prevent crashing the free server."""
        if self.model is not None:
            del self.model
            self.model = None
            tf.keras.backend.clear_session() # Tell TensorFlow to let go of memory
            gc.collect() # Force Python to clean up the trash

    def predict(self, processed_image):
        # 1. Load into memory
        self.load_model()
        
        # 2. Prepare image
        if processed_image.ndim == 3:
            img_array = np.expand_dims(processed_image, axis=0)
        else:
            img_array = processed_image
            
        # 3. Predict
        predictions = self.model.predict(img_array)
        probabilities = predictions[0] 
        
        # 4. THE TRANSLATOR FIX
        # If the AI outputs 1 number (Tumor Model - Sigmoid)
        if len(probabilities) == 1:
            ai_score = float(probabilities[0])
            if ai_score > 0.5:
                class_index = 1 # "yes"
                confidence = ai_score
            else:
                class_index = 0 # "no"
                confidence = 1.0 - ai_score # Flips a 0.1 score to 90% confident it's a "no"
                
        # If the AI outputs multiple numbers (Alzheimer's Model - Softmax)
        else:
            class_index = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
        
        # 5. Instantly delete from memory!
        self.unload_model()
        
        return {
            "label": self.labels[class_index],
            "confidence_score": confidence
        }
        
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
# 3. ALZHEIMER'S ENDPOINT
# ==========================================
ALZ_LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
ALZ_MODEL_PATH = "models/trained_alzheimers_model.keras"

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
TUMOR_MODEL_PATH = "models/trained_tumor_model.keras"

tumor_classifier = MRIClassifier(TUMOR_MODEL_PATH, TUMOR_LABELS)

@app.post("/predict/tumor")
async def predict_tumor(file: UploadFile = File(...)):
    image_bytes = await file.read()
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

# ==========================================
# 5. COMPREHENSIVE SCAN ENDPOINT
# ==========================================
@app.post("/predict/comprehensive")
async def predict_comprehensive(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        processed_image = processor.preprocess(image_bytes)
        
        # Because of our new class, it will load Model 1 -> Predict -> Unload Model 1
        alz_result = alzheimer_classifier.predict(processed_image)
        
        # Then it will load Model 2 -> Predict -> Unload Model 2
        tumor_result = tumor_classifier.predict(processed_image)
        
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