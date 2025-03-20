import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Define the sampling function for the Lambda layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 128), mean=0., stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Rebuild the EXACT encoder architecture from training
IMG_SHAPE = (224, 224, 3)
LATENT_DIM = 128

inputs = Input(shape=IMG_SHAPE)
x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
x = Flatten()(x)

z_mean = Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = Dense(LATENT_DIM, name="z_log_var")(x)
z = Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# ✅ Load weights (now matches the correct architecture)
encoder.load_weights("encoder_weights.h5")

# ⚡ Load the decoder normally
decoder = tf.keras.models.load_model("decoder.h5", compile=False)

# Preprocess an image
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

tum_sample = preprocess_img(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\Kaggle\colon_adeno\colonca1.jpeg")

# Encode image
z_mean, z_log_var, z = encoder.predict(tum_sample)

# Modify latent vector (Shift towards NORM characteristics)
alpha = 0.7
norm_latent_mean = np.load("norm_latent_mean.npy")  # Load healthy latent space mean
transformed_z = alpha * norm_latent_mean + (1 - alpha) * z

# Decode to get the transformed image
transformed_img = decoder.predict(transformed_z)

# Display the transformed image
plt.imshow(transformed_img[0])
plt.axis('off')
plt.show()
