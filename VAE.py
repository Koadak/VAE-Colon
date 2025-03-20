import tensorflow as tf
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from vae_model import create_vae  # Import updated VAE model

# GPU check
print(tf.config.list_physical_devices('GPU'))

# Paths to datasets
# norm_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM")
tum_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\TUM")

# Load images
# norm_images = list(norm_path.glob("*.png"))
tum_images = list(tum_path.glob("*.png"))

# all_images = norm_images + tum_images  # Combine both classes
# all_labels = [0] * len(norm_images) + [1] * len(tum_images)  # Labels (0 = NORM, 1 = TUM)

all_images = tum_images  # Combine both classes
all_labels = [1] * len(tum_images)  # Labels (0 = NORM, 1 = TUM)

# Convert paths to strings
all_image_paths = [str(path) for path in all_images]

# Split into train (80%) and test (20%)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Image processing parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 24

# Function to process images
def process_path(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)  # Convert to RGB
    image = tf.image.resize(image, IMG_SIZE)  # Resize
    image = image / 255.0  # Normalize
    return image, label

# Create train and test datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Apply processing
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create and compile VAE model with the frozen classifier
vae = create_vae()
vae.compile(optimizer=tf.keras.optimizers.Adam())

# Train VAE
vae.fit(train_dataset, epochs=5, validation_data=test_dataset)

# Save trained models
vae.encoder.save("encoder.h5")
vae.encoder.save_weights("encoder_weights.h5")
vae.decoder.save("decoder.h5")
vae.save_weights("vae_weights.h5")

# Extract normal images from the test dataset for evaluation
norm_test_paths = [p for p, label in zip(test_paths, test_labels) if label == 0]
