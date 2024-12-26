import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras import Model,ops
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda,Layer
from keras.optimizers import Adam

def sample_point_from_normal_distribution(args):
    mu, log_variance = args
    epsilon = tf.random.normal(shape=tf.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.exp(tf.clip_by_value(log_variance, -10, 10) / 2) * epsilon
    return random_sample

class VAEModel(Model):
    def __init__(self, encoder, decoder,reconstruction_loss_weight, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z,z_mean, z_log_var = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss= ops.mean(
                ops.square(data - reconstruction), 
                axis=[1, 2, 3],
            )* self.reconstruction_loss_weight
            kl_loss = -0.5 * ops.sum(
                1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis=1
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def test_step(self, data):
        z,z_mean, z_log_var = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss= ops.mean(
            ops.square(data - reconstruction), 
            axis=[1, 2, 3],
        )* self.reconstruction_loss_weight
        kl_loss = -0.5 * ops.sum(
            1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
            axis=1
        )
        total_loss = reconstruction_loss + kl_loss
        # Update trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return metrics
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class VAE:

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = tf.cast(tf.reduce_prod(input_shape), tf.float32)

        self.encoder = None
        self.decoder = None
        self.model = None
        self._num_conv_layers = len(conv_filters)

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        # self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=None)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations,_,_ = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "checkpoint.weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "checkpoint.weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self.model, self.encoder, self.decoder = self._build_autoencoder(
            input_shape=self.input_shape,
            conv_filters=self.conv_filters,
            conv_kernels=self.conv_kernels,
            conv_strides=self.conv_strides,
            latent_space_dim=self.latent_space_dim
        )

    def _build_encoder(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        encoder_input = Input(shape=input_shape, name="encoder_input")
        x = encoder_input
        for layer_index in range(len(conv_filters)):
            x = Conv2D(
                filters=conv_filters[layer_index],
                kernel_size=conv_kernels[layer_index],
                strides=conv_strides[layer_index],
                padding="same",
                name=f"encoder_conv_layer_{layer_index + 1}"
            )(x)
            x = ReLU(name=f"encoder_relu_{layer_index + 1}")(x)
            x = BatchNormalization(momentum=0.9, name=f"encoder_bn_{layer_index + 1}")(x)

        shape_before_bottleneck = x.shape[1:]
        x = Flatten()(x)
        mu = Dense(latent_space_dim, name="mu")(x)
        log_variance = Dense(latent_space_dim, name="log_variance")(x)
        encoder_output = Lambda(sample_point_from_normal_distribution, output_shape=(latent_space_dim,), name="encoder_output")([mu, log_variance])

        encoder = Model(encoder_input, [encoder_output,mu,log_variance], name="encoder")
        return encoder, shape_before_bottleneck

    def _build_decoder(self, latent_space_dim, shape_before_bottleneck, conv_filters, conv_kernels, conv_strides):
        decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")
        x = Dense(np.prod(shape_before_bottleneck), name="decoder_dense")(decoder_input)
        x = Reshape(shape_before_bottleneck)(x)

        for layer_index in reversed(range(1, len(conv_filters))):
            x = Conv2DTranspose(
                filters=conv_filters[layer_index],
                kernel_size=conv_kernels[layer_index],
                strides=conv_strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{len(conv_filters) - layer_index}"
            )(x)
            x = ReLU(name=f"decoder_relu_{len(conv_filters) - layer_index}")(x)
            x = BatchNormalization(momentum=0.9, name=f"decoder_bn_{len(conv_filters) - layer_index}")(x)

        x = Conv2DTranspose(
            filters=1,
            kernel_size=conv_kernels[0],
            strides=conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{len(conv_filters)}"
        )(x)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)

        decoder = Model(decoder_input, decoder_output, name="decoder")
        return decoder

    def _build_autoencoder(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        encoder, shape_before_bottleneck = self._build_encoder(
            input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim)
        decoder = self._build_decoder(latent_space_dim, shape_before_bottleneck, conv_filters, conv_kernels, conv_strides)
        # Build the model
        model = VAEModel(encoder, decoder,self.reconstruction_loss_weight, name="autoencoder")
        return model, encoder, decoder

if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2),
        latent_space_dim=2
    )
    autoencoder.summary()
