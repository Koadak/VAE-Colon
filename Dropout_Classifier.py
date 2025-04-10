import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np

class GaussianBlurLayer(Layer):
    def __init__(self, kernel_size=5, sigma=1.0, **kwargs):
        super(GaussianBlurLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        def gaussian_kernel(size, sigma):
            """Generates a 2D Gaussian kernel."""
            ax = np.arange(-size // 2 + 1., size // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            kernel = kernel / np.sum(kernel)
            return kernel.astype(np.float32)

        kernel = gaussian_kernel(self.kernel_size, self.sigma)
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        kernel = np.repeat(kernel, 3, axis=2)  # Apply same kernel to R, G, B
        kernel = np.repeat(kernel, 3, axis=3)  # Each input channel to each output

        self.kernel = tf.constant(kernel, dtype=tf.float32)
        self.kernel = tf.transpose(self.kernel, [2, 3, 0, 1])  # shape: [in, out, k, k]
        super(GaussianBlurLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Apply depthwise convolution (blurring)
            return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return inputs

    def get_config(self):
        config = super(GaussianBlurLayer, self).get_config()
        config.update({"kernel_size": self.kernel_size, "sigma": self.sigma})
        return config




import tensorflow as tf
import keras
print(tf.config.list_physical_devices('GPU'))


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import load_model

# Load original model
original_model = load_model(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\74_class_model.h5")

# Start with the input
inputs = Input(shape=(224, 224, 3))  # use (None, None, 3) if dynamic shape is needed

# Rebuild layer-by-layer and insert Dropout
x = inputs
for layer in original_model.layers[1:-2]:  # exclude Input and final Dense
    x = layer(x)
    if "pool" in layer.name:
        x = Dropout(0.2)(x)  # Dropout after pooling layers

# Add global pooling (if needed)
x = original_model.layers[-2](x)  # global_average_pooling2d_1

# Add dropout before final dense layer
x = Dropout(0.2)(x)

# Final output
outputs = Dense(1, activation='sigmoid', name='binary_output')(x)

# Create new model
dropout_model = Model(inputs=inputs, outputs=outputs)

# Compile
dropout_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

dropout_model.summary()



for i, layer in enumerate(dropout_model.layers):
    print(layer.trainable)
# Verify
# new_model.summary()

import tensorflow as tf
import pathlib

norm_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM")
tum_path = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\TUM")

# Get list of image paths
norm_images = list(norm_path.glob("*.png"))[:400]
tum_images = list(tum_path.glob("*.png"))[:400]

# norm_path_2 = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\NORM_ADV")
# tum_path_2 = pathlib.Path(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Test_Train\TUM_ADV")
#
# # Get list of image paths
# norm_images_2 = list(norm_path.glob("*.png"))
# tum_images_2 = list(tum_path.glob("*.png"))


total_norm = norm_images #+ norm_images_2
total_tum = tum_images #+ tum_images_2


# Create labels (0 for NORM, 1 for TUM)
norm_labels = [0] * len(total_norm)
tum_labels = [1] * len(total_tum)

# Combine file paths and labels
all_images = total_norm + total_tum
all_labels = norm_labels + tum_labels


# print(len (norm_images))
# print(len (tum_images))

import tensorflow as tf
import pathlib

# Define image size and batch size
IMG_SIZE = (224, 224)  # Modify based on your model requirements
BATCH_SIZE = 25

# Convert paths to strings
all_image_paths = [str(path) for path in all_images]

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))


def process_path(file_path, label):
    # Read image
    image = tf.io.read_file(file_path)
    # Decode PNG
    image = tf.image.decode_png(image, channels=3)  # Convert to RGB
    # Resize image
    image = tf.image.resize(image, IMG_SIZE)
    # Normalize pixel values (optional)
    image = image / 255.0  # Scale to [0,1]
    return image, label

# Apply the function to dataset
dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, and prefetch
dataset = dataset.shuffle(buffer_size=1400).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)



# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size  # Remaining 10%

# Split dataset
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)  # Remaining for test


print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset  ))


with tf.device("/GPU:0"):
    history = dropout_model.fit(train_dataset, validation_data=val_dataset, epochs=20)


loss, accuracy = dropout_model.evaluate(test_dataset)
print(f"Validation Accuracy: {accuracy:.2f}")

dropout_model.save('adeno_tester_dropout.h5')

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()