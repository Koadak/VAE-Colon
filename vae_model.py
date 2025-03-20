import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

IMG_SHAPE = (224, 224, 3)
LATENT_DIM = 128

# Load the pre-trained classifier and freeze its layers
classifier = load_model(r"C:\Users\jgdga\PycharmProjects\GPU_Tester\adeno_tester_1.h5")
classifier.trainable = False


# Encoder

def build_encoder():
    inputs = Input(shape=IMG_SHAPE)
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)

    z_mean = Dense(LATENT_DIM, name="z_mean")(x)
    # z_log_var = Dense(LATENT_DIM, name="z_log_var")(x)
    z_log_var = Dense(LATENT_DIM, name="z_log_var")(x)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)  # Avoid extreme values

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(LATENT_DIM,), name="z")([z_mean, z_log_var])

    return Model(inputs, [z_mean, z_log_var, z], name="encoder")


def sinkhorn_distance(z_true, z_pred, epsilon=0.01, num_iters=50):
    cost_matrix = tf.norm(tf.expand_dims(z_true, 1) - tf.expand_dims(z_pred, 0), axis=-1)

    # Small epsilon to prevent log(0)
    epsilon = tf.maximum(epsilon, 1e-8)

    u = tf.zeros_like(cost_matrix)
    v = tf.zeros_like(cost_matrix)

    for _ in range(num_iters):
        u = -epsilon * tf.math.log(tf.reduce_mean(tf.exp(-cost_matrix / epsilon + v[:, None]), axis=1) + 1e-8)
        v = -epsilon * tf.math.log(tf.reduce_mean(tf.exp(-cost_matrix / epsilon + u[None, :]), axis=0) + 1e-8)

    sinkhorn_dist = tf.reduce_mean(cost_matrix * tf.exp(-cost_matrix / epsilon + u[:, None] + v[None, :]))
    return sinkhorn_dist




# Decoder
def build_decoder():
    latent_inputs = Input(shape=(LATENT_DIM,))
    x = Dense(28 * 28 * 128, activation='relu')(latent_inputs)
    x = Reshape((28, 28, 128))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(latent_inputs, x, name="decoder")


# Custom VAE model with classifier integration
class VAE(Model):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier  # Pre-trained classifier
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed



    def train_step(self, data):
        images, labels = data  # Labels are used for classification loss

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(images)
            reconstructed = self.decoder(z)

            # Reconstruction Loss (MSE)
            recon_loss = -1 * self.mse_loss(images, reconstructed)

            # Earth Mover's Distance (Sinkhorn)
            z_true = K.random_normal(shape=K.shape(z_mean))  # Sample from prior N(0,1)
            emd_loss = sinkhorn_distance(z_true, z_mean)

            # Classification Loss (Use the classifier to predict class)
            class_preds = self.classifier(images, training=False)  # Get classifier predictions

            # 🔹 Reshape class_preds to match labels
            class_preds = tf.reshape(class_preds, tf.shape(class_preds))


            class_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, class_preds, from_logits=True)
            class_loss = tf.reduce_mean(class_loss)  # Take mean over batch

            # Total loss (Reconstruction + EMD + Classification)
            total_loss = recon_loss + emd_loss + class_loss

        # grads = tape.gradient(total_loss, self.trainable_variables)
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]  # Clip gradients to avoid NaNs
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "emd_loss": emd_loss,
            "classification_loss": class_loss
        }


# Function to create the VAE
def create_vae():
    encoder = build_encoder()
    decoder = build_decoder()
    vae = VAE(encoder, decoder, classifier)
    return vae
