import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# === Load the classifier model ===
classifier = tf.keras.models.load_model("adeno_tester_1.h5", compile=False)

# === Image preprocessing parameters ===
IMG_SIZE = (224, 224)

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# === Generate adversarial example targeting the classifier ===
def generate_adversarial_example_until_flip(classifier, input_image, true_label, epsilon=0.01, max_steps=100):
    x_adv = tf.Variable(input_image)
    true_label = tf.convert_to_tensor([[true_label]], dtype=tf.float32)
    flipped_label = 1.0 - true_label
    optimizer = tf.keras.optimizers.Adam(learning_rate=epsilon)

    for step in range(max_steps):
        with tf.GradientTape() as tape:
            class_probs = classifier(x_adv, training=False)
            loss = tf.keras.losses.binary_crossentropy(flipped_label, class_probs)

        preds = class_probs.numpy()
        if ((preds >= 0.5) != bool(true_label.numpy())):  # Class has flipped
            print(f"Prediction flipped at step {step} (prob: {preds[0][0]:.4f})")
            break

        grads = tape.gradient(loss, x_adv)
        optimizer.apply_gradients([(grads, x_adv)])
        x_adv.assign(tf.clip_by_value(x_adv, 0.0, 1.0))  # Keep image valid

    return x_adv.numpy()[0], step + 1



# === Helper to print classifier prediction ===
def print_prediction_info(prob, label=""):
    prob = prob.numpy().squeeze()
    cls = "TUM" if prob >= 0.5 else "NORM"
    print(f"{label} classifier probability: {prob:.4f} --> Class: {cls}")
    return cls

# === Path to your test image ===
image_path = r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Kaggle\colon_adeno\colonca14.jpeg"

# === Load and preprocess image ===
image = load_and_preprocess_image(image_path)
input_image = tf.expand_dims(image, axis=0)

# === Get prediction on original ===
original_pred = classifier(input_image, training=False)

# === Generate adversarial image ===
true_label = 1  # 0 for NORM, 1 for TUM
adv_image, steps_taken = generate_adversarial_example_until_flip(classifier, input_image, true_label)
print(f"Adversarial image generated in {steps_taken} steps.")

adv_input_tensor = tf.expand_dims(adv_image, axis=0)
adv_pred = classifier(adv_input_tensor, training=False)

# === Display predictions ===
print()
orig_cls = print_prediction_info(original_pred, "Original input")
adv_cls = print_prediction_info(adv_pred, "Adversarial input")

# === Visualize results ===
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.title(f"Original Input ({orig_cls})")
plt.imshow(image.numpy())
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Adversarial Input ({adv_cls})")
plt.imshow(adv_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Perturbation ×100")
diff = np.abs(image.numpy() - adv_image) * 100
plt.imshow(diff)
plt.axis("off")

plt.tight_layout()
plt.show()


print()
l2_dist = np.linalg.norm((adv_image - image.numpy()).flatten())
linf_dist = np.max(np.abs(adv_image - image.numpy()))
mad = np.mean(np.abs(adv_image - image.numpy()))

print(f"L2 distance between original and adversarial image: {l2_dist:.6f}")
print(f"L∞ (max) perturbation: {linf_dist:.6f}")
print(f"Mean Absolute Pixel Difference: {mad:.6f}")


perturbation = np.abs(adv_image - image.numpy())  # Shape: (224, 224, 3)
heatmap = np.mean(perturbation, axis=-1)  # Average over channels for 2D view

# plt.figure(figsize=(6, 5))
# plt.title("Perturbation Heatmap (Mean over RGB)")
# plt.imshow(heatmap, cmap='hot')
# plt.colorbar(label="Pixel-wise difference")
# plt.axis("off")
# plt.show()
