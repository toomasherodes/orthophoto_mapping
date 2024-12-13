import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load .tif image (both orthophoto and mask)
def load_tif(filepath):
    with Image.open(filepath) as img:
        return np.array(img)

# Preprocess images: normalize orthophoto, binarize mask
def preprocess_images(orthophoto, mask):
    # Normalize orthophoto to range [0, 1]
    orthophoto_normalized = orthophoto / 255.0
    
    # Convert the 3-channel mask to a binary mask (0 for background, 1 for house)
    # This assumes the mask has [255, 255, 255] for house and [0, 0, 0] for background
    mask_binary = np.all(mask == [255, 255, 255], axis=-1).astype(np.float32)  # Check if all channels are [255, 255, 255]
    
    # Convert to (500, 500, 1)
    mask_binary = np.expand_dims(mask_binary, axis=-1)  # Ensure shape is (500, 500, 1)
    
    return orthophoto_normalized, mask_binary


# Generator for loading images and masks
def data_generator(image_paths, mask_paths, target_size):
    for img_path, mask_path in zip(image_paths, mask_paths):
        orthophoto = load_tif(img_path)
        mask = load_tif(mask_path)
        # Preprocess the images
        orthophoto_preprocessed, mask_preprocessed = preprocess_images(orthophoto, mask)
        yield orthophoto_preprocessed, mask_preprocessed

# Create tf.data.Dataset from the generator
def create_dataset(image_paths, mask_paths, target_size, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_paths, mask_paths, target_size),
        output_signature=(
            tf.TensorSpec(shape=target_size + (3,), dtype=tf.float32),  # Orthophoto (3 channels)
            tf.TensorSpec(shape=target_size + (1,), dtype=tf.float32),  # Mask (binary)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

import tensorflow as tf

def custom_cnn_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder part
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder part (upsample)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Final output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Crop the output to match the target size
    outputs_cropped = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(outputs)

    model = tf.keras.models.Model(inputs, outputs_cropped)
    return model


# Helper function to load all image paths from a directory
def load_image_paths(directory):
    return [os.path.join(directory, filename) for filename in sorted(os.listdir(directory)) if filename.endswith('.tif')]

# Set paths to directories
train_image_paths = load_image_paths("./train/img")
train_mask_paths = load_image_paths("./train/mask")
val_image_paths = load_image_paths("./val/img")
val_mask_paths = load_image_paths("./val/mask")

# Create datasets
target_size = (500, 500)  # Update to 500x500
batch_size = 16
train_dataset = create_dataset(train_image_paths, train_mask_paths, target_size, batch_size)
val_dataset = create_dataset(val_image_paths, val_mask_paths, target_size, batch_size)

# Define and compile the custom CNN model
input_shape = target_size + (3,)
model = custom_cnn_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Load and preprocess a single image from the validation set
def load_and_preprocess_sample(image_paths, mask_paths, idx, target_size):
    # Load the image and mask using the index
    orthophoto = load_tif(image_paths[idx])
    mask = load_tif(mask_paths[idx])

    # Preprocess the image and mask
    orthophoto_preprocessed, mask_preprocessed = preprocess_images(orthophoto, mask)
    
    # Resize if needed
    orthophoto_resized = tf.image.resize(orthophoto_preprocessed, target_size)
    mask_resized = tf.image.resize(mask_preprocessed, target_size)
    
    return orthophoto_resized.numpy(), mask_resized.numpy()

# Example: Visualizing the first image from the validation set
idx = 0  # Change this index to visualize a different image
orthophoto, mask = load_and_preprocess_sample(val_image_paths, val_mask_paths, idx, target_size)

# Make a prediction with the model
prediction = model.predict(tf.expand_dims(orthophoto, axis=0))[0]

# Visualize the orthophoto, ground truth mask, and predicted mask
def visualize_prediction(model, orthophoto, mask):
    prediction = model.predict(tf.expand_dims(orthophoto, axis=0))[0]
    plt.figure(figsize=(12, 4))
    
    # Orthophoto
    plt.subplot(1, 3, 1)
    plt.title("Orthophoto")
    plt.imshow(orthophoto)
    plt.axis('off')
    
    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis('off')
    
    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction.squeeze(), cmap="gray")
    plt.axis('off')
    
    plt.show()

# Visualize the first image from the validation set
visualize_prediction(model, orthophoto, mask)
