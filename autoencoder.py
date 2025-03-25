import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.models import load_model  # Changed import
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# GPU configuration (add at start)
gpus = tf.config.list_physical_devices('GPU')


# Define dataset paths
norm_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM")
tum_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\TUM")

# Load image paths and assign labels (0 = NORM, 1 = TUM)
norm_images = list(norm_path.glob("*.png"))
tum_images = list(tum_path.glob("*.png"))
all_images = norm_images + tum_images
all_labels = [0] * len(norm_images) + [1] * len(tum_images)
all_image_paths = [str(path) for path in all_images]

# Split into train/test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Image processing parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 8


# Revised dataset processing function without extra batch dimension
def process_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return image, label


# Create targets as a tuple: (reconstruction_target, label_target)
def create_autoencoder_target(image, label):
    # Both the input and the reconstruction target are the same image.
    return image, (image, label)


# Create datasets (no unbatching needed)
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

train_dataset = (train_dataset
                 .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .map(create_autoencoder_target)
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
                .map(create_autoencoder_target)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

# Verify dataset shapes
print("\nDataset structure verification:")
sample = next(iter(train_dataset))
print("Input shape:", sample[0].shape)  # Expected: (8, 224, 224, 3)
print("Target tuple shapes:")
print("- Image target:", sample[1][0].shape)  # Expected: (8, 224, 224, 3)
print("- Label target:", sample[1][1].shape)  # Expected: (8,)

# Load classifier model
classifier = load_model(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\adeno_tester_1.h5")

# Extract feature extractor from classifier
# Adjust the layer name below to match a valid layer in your classifier model.
feature_extractor = tf.keras.Model(
    inputs=classifier.input,
    outputs=classifier.get_layer("block4_conv3").output
)


# Loss functions remain as before
def perceptual_loss(y_true, y_pred):
    true_features = feature_extractor(y_true, training=False)
    pred_features = feature_extractor(y_pred, training=False)
    return tf.reduce_mean(tf.abs(true_features - pred_features))


def classification_loss(true_label, y_pred):
    fake_class_probs = classifier(y_pred, training=False)
    flipped_labels = tf.expand_dims(1 - true_label, axis=-1)  # Expand dims to match classifier output shape
    per_sample_loss = tf.keras.losses.binary_crossentropy(flipped_labels, fake_class_probs)
    return tf.reduce_mean(per_sample_loss)  # Reduce to a scalar




def build_autoencoder():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    # Encoder
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs, decoded)


def build_autoencoder_modified(): # Using residual skip connections
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))


    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)
    x1p = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x1)

    x2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x1p)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.1)(x2)
    x2p = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x2)

    x3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x2p)
    x3 = tf.keras.layers.LeakyReLU(alpha=0.1)(x3)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x3)


    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(encoded)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x3])  # Skip connection

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x2])  # Skip connection

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, x1])  # Skip connection

    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs, decoded)


def build_non_bottleneck_autoencoder():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    # Encoder-like layers (but not really compressing)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder-like layers (mirroring encoder)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs, decoded)


autoencoder = build_autoencoder_modified()


class CustomAutoencoder(tf.keras.Model):
    def __init__(self, autoencoder, lambda_a=0.1, **kwargs):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.lambda_a = lambda_a

    def train_step(self, data):
        # data is a tuple: (inputs, (image_target, label_target))
        x, (image_target, label_target) = data

        with tf.GradientTape() as tape:
            y_pred = self.autoencoder(x, training=True)
            p_loss = perceptual_loss(image_target, y_pred)
            c_loss = classification_loss(label_target, y_pred)
            loss = self.lambda_a * p_loss + (1 - self.lambda_a) * c_loss

        gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))

        # Compute MSE manually as a scalar
        mse = tf.reduce_mean(tf.square(image_target - y_pred))
        # Return metrics without calling compiled metrics update
        return {"mse": mse, "loss": loss}

    def test_step(self, data):
        x, (image_target, label_target) = data
        y_pred = self.autoencoder(x, training=False)
        p_loss = perceptual_loss(image_target, y_pred)
        c_loss = classification_loss(label_target, y_pred)
        loss = self.lambda_a * p_loss + (1 - self.lambda_a) * c_loss

        mse = tf.reduce_mean(tf.square(image_target - y_pred))
        return {"mse": mse, "loss": loss}

    def call(self, inputs, training=False):
        return self.autoencoder(inputs, training=training)



# Instantiate and compile the custom model
custom_autoencoder = CustomAutoencoder(autoencoder, lambda_a=0.35)
custom_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[]  # Remove metrics to avoid built-in metric state updates
)



# Model summary verification
print("\nAutoencoder structure:")
autoencoder.summary()

# Train the custom autoencoder
EPOCHS = 30
with tf.device("/GPU:0"):
    history = custom_autoencoder.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        verbose=1
    )

# Save the trained autoencoder
custom_autoencoder.autoencoder.save("autoencoder_flipped_labels_3.h5")

# Optional: Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
