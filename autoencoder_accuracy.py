import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tqdm import tqdm

# Paths
norm_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM")
tum_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\TUM")

# Load models
autoencoder = tf.keras.models.load_model("autoencoder_flipped_labels_with_bottleneck_3.h5", compile=False)
classifier = tf.keras.models.load_model("adeno_tester_1.h5", compile=False)

# Parameters
IMG_SIZE = (224, 224)
threshold = 0.5  # classifier threshold for TUM vs NORM

# Image loading function
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Evaluation loop
def evaluate_folder(folder_path, true_label):
    results = []
    image_paths = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg"))[:800]

    for path in tqdm(image_paths, desc=f"Evaluating {'NORM' if true_label == 0 else 'TUM'}"):
        image = load_and_preprocess_image(path)
        input_image = tf.expand_dims(image, axis=0)

        # Reconstruct
        reconstructed = autoencoder(input_image, training=False)

        # Predict
        original_prob = classifier(input_image, training=False).numpy().squeeze()
        reconstructed_prob = classifier(reconstructed, training=False).numpy().squeeze()

        original_pred = 1 if original_prob >= threshold else 0
        reconstructed_pred = 1 if reconstructed_prob >= threshold else 0

        # Store result
        results.append({
            "filename": str(path.name),
            "true_label": true_label,
            "original_prob": original_prob,
            "reconstructed_prob": reconstructed_prob,
            "original_pred": original_pred,
            "reconstructed_pred": reconstructed_pred
        })

    return results

# Run evaluation
norm_results = evaluate_folder(norm_path, true_label=0)  # 0 = NORM
tum_results = evaluate_folder(tum_path, true_label=1)    # 1 = TUM

# Combine and compute metrics
all_results = norm_results + tum_results

# Measure performance
flip_success_count = 0
total = len(all_results)

for res in all_results:
    if res["original_pred"] != res["reconstructed_pred"]:
        flip_success_count += 1

flip_accuracy = flip_success_count / total
print(f"\nFlip Success Rate: {flip_accuracy:.2%} ({flip_success_count}/{total})")