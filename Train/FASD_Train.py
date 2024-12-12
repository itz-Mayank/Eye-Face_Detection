import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# Set GPU 0 as the visible device
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use GPU 0
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU 0:", gpus[0])
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
else:
    print("No GPUs found. Ensure proper driver and CUDA installation.")

# Enable mixed precision training for faster computation
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 45
REAL_IMG_DIR = "LCC_FASD_training/real"
SPOOF_IMG_DIR = "LCC_FASD_training/spoof"

# Collect real and spoof images
real_images = [os.path.join(REAL_IMG_DIR, img) for img in os.listdir(REAL_IMG_DIR) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
spoof_images = [os.path.join(SPOOF_IMG_DIR, img) for img in os.listdir(SPOOF_IMG_DIR) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

# Create DataFrame for real and spoof images
real_df = pd.DataFrame({"image_path": real_images, "face_liveness": 1, "eye_liveness": 1})
spoof_df = pd.DataFrame({"image_path": spoof_images, "face_liveness": 0, "eye_liveness": 0})

# Combine and shuffle the dataset
combined_df = pd.concat([real_df, spoof_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train/validation split
train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Data generator function
def create_generator(df, batch_size, img_height, img_width):
    while True:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            images, labels_face, labels_eye = [], [], []
            for _, row in batch.iterrows():
                img_path = row['image_path']
                try:
                    img = load_img(img_path, target_size=(img_height, img_width))
                    img = img_to_array(img) / 255.0
                    images.append(img)
                    labels_face.append(row['face_liveness'])
                    labels_eye.append(row['eye_liveness'])
                except Exception as e:
                    print(f"Skipping invalid image {img_path}: {e}")
            if len(images) > 0:
                yield np.array(images), [np.array(labels_face), np.array(labels_eye)]

# Initialize train and validation generators
train_generator = create_generator(train_df, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
val_generator = create_generator(val_df, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

# Define the multitask model
def build_model():
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Face liveness head
    face_liveness_head = Dense(1, activation="sigmoid", name="face_liveness")(x)
    # Eye liveness head
    eye_liveness_head = Dense(1, activation="sigmoid", name="eye_liveness")(x)

    return Model(inputs=base_model.input, outputs=[face_liveness_head, eye_liveness_head])

# Build and compile the model
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss={
        "face_liveness": "binary_crossentropy",
        "eye_liveness": "binary_crossentropy",
    },
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(val_df) // BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Save the trained model
model.save("face_eye_liveness_model_lccfasd.h5")
print("[INFO] Model saved as face_eye_liveness_model_lccfasd.h5")
