import tensorflow as tf
from tensorflow.keras import layers, models
import os

TRAIN_DIR = "data/alzheimers/data/train"
VAL_DIR = "data/alzheimers/data/val"

print("Loading data and enforcing Class Weights...")

# 1. Load the pre-split data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale"
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale"
)

class_names = train_dataset.class_names
print(f"Categories: {class_names}")

# 2. Normalize pixels
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# 3. Build the Neural Network
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- THE CLINICAL FIX: CLASS WEIGHTS ---
# Based on the exact image counts in your Kaggle dataset:
# Index 0: Mild Demented (896 images) -> 3.5x penalty
# Index 1: Moderate Demented (64 images) -> 50.0x penalty!
# Index 2: Non Demented (3200 images) -> 1.0x baseline penalty
# Index 3: Very Mild Demented (2240 images) -> 1.4x penalty
disease_weights = {
    0: 3.5,
    1: 50.0,
    2: 1.0,
    3: 1.4
}

print("Starting the training process with weighted penalties...")

# 4. Train the AI using the new weights
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=10,
    class_weight=disease_weights # <--- We inject the penalties here!
)

# 5. Save the smarter brain
os.makedirs("models", exist_ok=True)
model.save("models/trained_alzheimers_model.keras")

print("SUCCESS: Smarter, medically-balanced model saved!")