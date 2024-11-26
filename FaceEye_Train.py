import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Enable mixed precision training for faster computation
from tensorflow.keras import mixed_precision

# Set the policy for mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5  # Increased epochs as requested
ANNOTATION_FILE = "CelebA/Anno/list_attr_celeba.txt"
REAL_IMG_DIR = "CelebA/Faces/Real"
SPOOF_IMG_DIR = "CelebA/Faces/Spoof"
LIMITED_REAL_DATA = 2000  # Adjust the number of real images for limited data

# Load annotations from CelebA dataset
df = pd.read_csv(ANNOTATION_FILE, sep='\s+', header=1)
df.reset_index(inplace=True)  # Reset index to make image names a column
df.rename(columns={"index": "image_name"}, inplace=True)

# Filter dataset for face liveness and eye liveness detection
df['face_liveness'] = df['Eyeglasses'].apply(lambda x: 1 if x == 1 else 0)
df['eye_liveness'] = df['Narrow_Eyes'].apply(lambda x: 1 if x == 0 else 0)  # Real eyes are open (0)

# Combine real and spoofed faces
real_df = df[df['face_liveness'] == 1]  # Real faces
spoof_df = df[df['face_liveness'] == 0]  # Fake faces

# Assign spoofed faces a "fake" label for both face and eye liveness
real_df['eye_liveness'] = 1  # Real faces have live eyes
spoof_df['eye_liveness'] = 0  # Fake faces have fake eyes

# Limit the real dataset
limited_real_df = real_df.sample(n=LIMITED_REAL_DATA, random_state=42)

# Combine the full spoof data with the limited real data
combined_df = pd.concat([limited_real_df, spoof_df])

# Train/validation split
train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Image Data Generators
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Data generators function
def create_generator(df, img_dir, batch_size, img_height, img_width):
    while True:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            images = []
            labels_face = []
            labels_eye = []
            for idx, row in batch.iterrows():
                img_path = os.path.join(img_dir, row['image_name'])
                try:
                    if os.path.exists(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = load_img(img_path, target_size=(img_height, img_width))
                        img = img_to_array(img)
                        images.append(img)
                        labels_face.append(row['face_liveness'])
                        labels_eye.append(row['eye_liveness'])
                except Exception as e:
                    print(f"Skipping invalid image {img_path}: {e}")
            
            if len(images) > 0:
                yield np.array(images), [np.array(labels_face), np.array(labels_eye)]
            else:
                print(f"Skipping empty batch starting at index {i}")  # Debugging empty batches

# Initialize train and validation data generators
train_generator = create_generator(train_df, REAL_IMG_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
val_generator = create_generator(val_df, REAL_IMG_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

# Define the multitask model
def build_model():
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    # Face liveness head
    face_liveness_head = Dense(1, activation="sigmoid", name="face_liveness")(x)
    # Eye liveness head
    eye_liveness_head = Dense(1, activation="sigmoid", name="eye_liveness")(x)

    return Model(inputs=base_model.input, outputs=[face_liveness_head, eye_liveness_head])

# Build and compile the model
model = build_model()
model.compile(
    optimizer=Adam(lr=0.0001),
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

# Save the model
model.save("face_eye_liveness_model.h5")
print("[INFO] Model saved as face_eye_liveness_model.h5")
