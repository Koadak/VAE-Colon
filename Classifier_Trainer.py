import tensorflow as tf
import keras
print(tf.config.list_physical_devices('GPU'))


from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\74_class_model.h5")

# # Track the last convolutional layer
# last_conv_index = -1
# for i, layer in enumerate(model.layers):
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         last_conv_index = i
#
# # Freeze all layers up to (but not including) the last conv layer
# for i, layer in enumerate(model.layers):
#     if i < last_conv_index:
#         layer.trainable = False
#     else:
#         layer.trainable = True

for i, layer in enumerate(model.layers):
    layer.trainable = False


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


# Identify the correct layer before the final output
x = model.layers[-2].output  # Get the second-to-last layer's output

# Add a new binary classification layer
new_output = Dense(1, activation="sigmoid", name="binary_output")(x)

# Create the modified model
new_model = Model(inputs=model.input, outputs=new_output)

# Compile
new_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

for i, layer in enumerate(new_model.layers):
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
    new_model.fit(train_dataset, validation_data=val_dataset, epochs=10)


loss, accuracy = new_model.evaluate(test_dataset)
print(f"Validation Accuracy: {accuracy:.2f}")

new_model.save('adeno_tester_3.h5')