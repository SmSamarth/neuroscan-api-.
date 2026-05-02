import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
# 1. Define where your data is
# IMPORTANT: If your 'yes' and 'no' folders are inside another folder 
# (like 'data/brain_tumor_dataset'), update this path to match!
DATA_DIRECTORY = "data/brain_tumor_dataset" if os.path.exists("data/brain_tumor_dataset") else "data"

print(f"Loading dataset from: {DATA_DIRECTORY}")

# 2. Load and prepare the data
# We split the data: 80% to train the AI, 20% to test its accuracy
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale" # Matches your (224, 224, 1) OpenCV pipeline exactly
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale"
)

# 3. Normalize pixel values (Standard AI requirement: values between 0 and 1)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

print("Building the neural network architecture...")

# 4. Build the Convolutional Neural Network (CNN)
# 4. Build the Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 1)),
    
    # --- NEW AUGMENTATION LAYERS GO HERE ---
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.1),     
    # ---------------------------------------
    
    # Feature extraction layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the math out for the final decision
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    
    # Final layer: 1 node outputting a percentage (0 = Healthy, 1 = Tumor)
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Starting the training process...")

# 5. Train the AI! (Epochs = how many times it loops through the whole dataset)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# 6. Save the trained brain
os.makedirs("models", exist_ok=True)
model.save("models/trained_tumor_model.keras")

print("SUCCESS: Model successfully trained and saved to models/trained_tumor_model.keras!")

# --- ADD THIS TO THE BOTTOM OF train.py ---
print("Drawing accuracy graph...")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('AI Learning Curve (Accuracy)')
plt.xlabel('Epoch (Study Session)')
plt.ylabel('Accuracy Score')
plt.legend(loc='lower right')

plt.savefig('learning_curve.png')
print("Graph saved as learning_curve.png! Open it to see how your AI learned.")