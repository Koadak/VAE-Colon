import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained autoencoder (no compile needed)
autoencoder = tf.keras.models.load_model(
    "autoencoder_flipped_labels_2.h5", compile=False
)

# Load the classifier
classifier = tf.keras.models.load_model(
    "adeno_tester_1.h5", compile=False
)

# Image processing parameters
IMG_SIZE = (224, 224)

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # JPEG-specific
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Path to your test JPEG image
image_path = r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM\NORM-TCGA-DSFGMFFQ.png"

# Load and preprocess image
image = load_and_preprocess_image(image_path)

# Add batch dimension
input_image = tf.expand_dims(image, axis=0)

# Autoencoder reconstruction
reconstructed = autoencoder(input_image, training=False)

# Classify original and reconstructed
original_pred = classifier(input_image, training=False)
reconstructed_pred = classifier(reconstructed, training=False)

# Convert to numpy for plotting and displaying results
input_np = image.numpy()
recon_np = tf.squeeze(reconstructed, axis=0).numpy()

# Convert classifier outputs to readable values
original_prob = original_pred.numpy().squeeze()
reconstructed_prob = reconstructed_pred.numpy().squeeze()


print(f"\nClassifier prediction on original image: {original_prob:.4f}")
print(f"Classifier prediction on reconstructed image: {reconstructed_prob:.4f}")

# Optional: threshold (you can tweak this)
threshold = 0.5
original_class = "TUM" if original_prob >= threshold else "NORM"
reconstructed_class = "TUM" if reconstructed_prob >= threshold else "NORM"

print(f"Original class prediction: {original_class}")
print(f"Reconstructed class prediction: {reconstructed_class}")

# Plot original and reconstructed
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title(f"Original ({original_class})")
plt.imshow(input_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Reconstructed ({reconstructed_class})")
plt.imshow(recon_np)
plt.axis("off")

plt.tight_layout()
plt.show()
