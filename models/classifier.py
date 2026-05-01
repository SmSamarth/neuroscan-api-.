import tensorflow as tf
import numpy as np

class MRIClassifier:
    def __init__(self, model_path, labels):
        # Load the specific model file
        self.model = tf.keras.models.load_model(model_path)
        self.labels = labels

    def predict(self, processed_image):
        # Ensure dimensions are (1, height, width, channels)
        img_array = np.expand_dims(processed_image, axis=0)
        
        # Get prediction from model
        predictions = self.model.predict(img_array)
        
        # Use the raw output (most .keras models have softmax built-in)
        probabilities = predictions[0]
        
        class_index = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            "label": self.labels[class_index],
            "confidence_score": confidence
        }